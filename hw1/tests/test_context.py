import unittest
from src.context import pack_context,block_ids,BasePromptTooLarge


class ContextPackingTests(unittest.TestCase):
    def test_skips_oversized_candidate_and_keeps_later_fitting_sources(self):
        question={'id':'q'}
        candidates=[{'chunk_id':'a','block_id':'A','text':'x'*80},
                    {'chunk_id':'b','block_id':'B','text':'y'*2},
                    {'chunk_id':'c','block_id':'C','text':'z'*2}]
        def build(q,chunks,no_context=False,concise_analysis=False):
            return [{'role':'user','content':''.join(c['text'] for c in chunks)}]
        kept,messages,skipped=pack_context(question,candidates,build,
            lambda messages:len(messages[0]['content']),10)
        self.assertEqual(block_ids(kept),['B','C'])
        self.assertEqual(len(skipped),1)
        self.assertEqual(skipped[0]['chunk_id'],'a')
        self.assertEqual(messages[0]['content'],'yyzz')

    def test_real_source_prompt_is_counted_as_a_whole(self):
        question={'id':'q','question':'Q','options':[{'id':'A','text':'option'}]}
        chunks=[{'chunk_id':'1','block_id':'source','page':1,'text':'a'*100}]
        def build(q,cs,no_context=False,concise_analysis=False):
            return [{'role':'system','content':'s'*20},
                    {'role':'user','content':q['question']+''.join(c['text'] for c in cs)}]
        kept,_,skipped=pack_context(question,chunks,build,
            lambda messages:sum(len(m['content']) for m in messages),50)
        self.assertEqual(kept,[])
        self.assertEqual(skipped[0]['reason'],'input_limit')

    def test_base_prompt_over_budget_fails_without_packing_candidates(self):
        def build(q,chunks,no_context=False,concise_analysis=False):
            return [{'role':'user','content':'x'*12}]
        with self.assertRaises(BasePromptTooLarge):
            pack_context({},[{'block_id':'never'}],build,lambda m:len(m[0]['content']),10)


if __name__=='__main__':unittest.main()
