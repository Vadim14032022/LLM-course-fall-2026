import json
import unittest
from unittest.mock import patch
from src.generation import generate_answer
from src.pipeline import messages_for, SYSTEM_NO_CONTEXT
from scripts.export_call_metadata import public_record


class Client:
    def __init__(self, final):
        self.final = final
        self.requests = []

    def tail_tokens(self, text, cap):
        return text[-cap:]

    def complete(self, qid, messages, max_tokens, budget, **kwargs):
        self.requests.append((messages, max_tokens))
        first = len(self.requests) == 1
        consumed = max_tokens if first else min(100, max_tokens)
        budget['llm_calls'] += 1
        budget['input_tokens'] += 300
        budget['output_tokens'] += consumed
        return ('' if first else self.final), {
            'reasoning_text':'Unfinished draft', 'usage':{'prompt_tokens':300,'completion_tokens':consumed},
            'max_tokens':max_tokens}


class GenerationTests(unittest.TestCase):
    def test_no_context_has_a_separate_unambiguous_prompt(self):
        prompt=messages_for({'question':'Q','options':[]},[],no_context=True)[0]['content']
        self.assertEqual(prompt,SYSTEM_NO_CONTEXT)
        self.assertIn('citations всегда []',prompt)
        self.assertNotIn('фрагменты учебника',prompt)

    def test_normalizer_does_not_control_retry_routing(self):
        client=Client('{}')
        budget={'llm_calls':0,'input_tokens':0,'output_tokens':0,'usage_verified':True}
        with patch('src.generation.parse_answer',return_value={'postprocessed':True}):
            generate_answer(client,{'id':'q','options':[{'id':'X'}]},[],['s'],
                {'response_strategy':'draft_then_prefill','max_tokens':4096},
                {'limits':{'max_output_tokens_per_question':4096}},budget,{})
        self.assertEqual(len(client.requests),2)

    def test_public_accounting_excludes_sources_and_reasoning(self):
        from src.validation import final_context
        record=public_record({'id':'q','context_block_ids':['source'],
            'context':[{'chunk_id':'chunk','block_id':'source','text':'PRIVATE SOURCE'}],
            'calls':[{'request_id':'api-id','reasoning_text':'PRIVATE THOUGHT',
                      'context_block_ids':['source'],'context_chunk_ids':['chunk'],
                      'response_text':'PRIVATE RESPONSE','usage':{'prompt_tokens':10,'completion_tokens':20}}]})
        self.assertEqual(record['calls'][0]['request_id'],'api-id')
        self.assertNotIn('PRIVATE',json.dumps(record))
        self.assertEqual(final_context(record), (['source'], []))

    def test_prefill_strategy_closes_draft_in_prompt_and_keeps_budget(self):
        answer = json.dumps({'answer_ids':['X'],'citations':['source'],'abstain':False,'rationale':'Supported'})
        client=Client(answer)
        budget={'llm_calls':0,'input_tokens':0,'output_tokens':0,'usage_verified':True}
        generate_answer(client, {'id':'unseen','options':[{'id':'X'}]},
            [{'role':'user','content':'Question'}], ['source'],
            {'response_strategy':'draft_then_prefill','draft_tokens':2048,'max_tokens':4096},
            {'limits':{'max_output_tokens_per_question':4096}},budget,{})
        continuation=client.requests[1][0][-1]
        self.assertEqual(continuation['role'],'assistant')
        self.assertEqual(continuation['content'],'{')
        self.assertEqual(continuation['reasoning_content'],'Unfinished draft')
        self.assertEqual(budget['output_tokens'],2148)

    def test_second_call_uses_only_remaining_budget_and_local_draft(self):
        answer = {'answer_ids':['X'],'citations':['source'],'abstain':False,'rationale':'Supported'}
        client = Client(json.dumps(answer))
        budget = {'llm_calls':0,'input_tokens':0,'output_tokens':0,'usage_verified':True}
        diagnostic = {}
        result = generate_answer(client, {'id':'unseen','options':[{'id':'X'}]},
            [{'role':'user','content':'Question'}], ['source'],
            {'response_strategy':'draft_then_finalize','draft_tokens':2048,'max_tokens':4096},
            {'limits':{'max_output_tokens_per_question':4096}}, budget, diagnostic)
        self.assertEqual(result['answer_ids'], ['X'])
        self.assertEqual([r[1] for r in client.requests], [2048,2048])
        self.assertEqual(budget['output_tokens'],2148)
        self.assertEqual(len(diagnostic['calls']),2)
        self.assertIn('Unfinished draft',client.requests[1][0][-2]['content'])

    def test_single_strategy_does_not_retry_after_exhausting_budget(self):
        client=Client('{}')
        budget={'llm_calls':0,'input_tokens':0,'output_tokens':0,'usage_verified':True}
        with self.assertRaises(ValueError):
            generate_answer(client, {'id':'unseen','options':[]}, [], [],
                {'max_tokens':4096}, {'limits':{'max_output_tokens_per_question':4096}},budget,{})
        self.assertEqual(len(client.requests),1)
        self.assertEqual(budget['output_tokens'],4096)

    def test_second_call_repacks_sources_against_remaining_request_budget(self):
        answer=json.dumps({'answer_ids':['X'],'citations':['small'],'abstain':False,'rationale':'Supported'})
        client=Client(answer)
        client.count=lambda messages,continue_final=False:sum(len(m.get('content',''))+len(m.get('reasoning_content','')) for m in messages)
        chunks=[{'chunk_id':'big','block_id':'big','page':1,'text':'x'*500},
                {'chunk_id':'small','block_id':'small','page':1,'text':'y'}]
        def build(q,selected,no_context=False,concise_analysis=False):
            return [{'role':'system','content':'s'*20},
                    {'role':'user','content':'question'+''.join(c['text'] for c in selected)}]
        budget={'llm_calls':0,'input_tokens':0,'output_tokens':0,'usage_verified':True}
        diagnostic={}
        result=generate_answer(client,{'id':'q','options':[{'id':'X'}]},build({'id':'q'},chunks),
            ['big','small'],{'response_strategy':'draft_then_finalize','draft_tokens':2048,'max_tokens':4096},
            {'limits':{'max_output_tokens_per_question':4096,'max_input_tokens_per_question':16384,
                       'max_input_plus_output_tokens_per_request':2400}},
            budget,diagnostic,context_chunks=chunks,message_builder=build,context_cap=3000)
        self.assertEqual(result['citations'],['small'])
        self.assertEqual(diagnostic['calls'][0]['context_block_ids'],['big','small'])
        self.assertEqual(diagnostic['calls'][1]['context_block_ids'],['small'])
        self.assertTrue(diagnostic['context_skipped_second_call'])
