import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validation import answer_errors, trace_errors, accounting_errors, final_context
from scripts.evaluate import evaluate


def accounting_fixture(n=2):
    calls = [{'usage':{'prompt_tokens':10,'completion_tokens':5},'max_tokens':32} for _ in range(n)]
    return ({'id':'q','llm_calls':n,'input_tokens':10*n,'output_tokens':5*n,
             'elapsed_ms':1,'usage_verified':True,'technical_error':False}, {'id':'q','calls':calls})


def evaluation_fixture():
    answers, gold, traces, questions = [], [], [], []
    for i,category in enumerate(['условия','сопоставление','применение','недостаточность данных']):
        qid=str(i); abstain=i==3
        questions.append({'id':qid,'question':'Question','options':[{'id':'A','text':'A'},{'id':'B','text':'B'}]})
        a={'id':qid,'answer_ids':[] if abstain else ['A'],'abstain':abstain,
           'citations':[] if abstain else ['s'],'rationale':'Explanation'}
        answers.append(a)
        gold.append(copy.deepcopy({**a,'category':category,'citation_sets':[[]] if abstain else [['s']]}))
        t,d=accounting_fixture(1);t['id']=qid;t['calls']=d['calls'];traces.append(t)
    return answers,gold,traces,questions


class ValidationTests(unittest.TestCase):
    def test_final_context_uses_sources_visible_to_last_call(self):
        answers,gold,traces,questions=evaluation_fixture()
        diagnostics=[]
        for trace in traces:
            call={**trace['calls'][0],'context_block_ids':['s'],'context_chunk_ids':['chunk:s']}
            diagnostics.append({'id':trace['id'],'context_block_ids':['s','extra'],'calls':[call]})
        result=evaluate(answers,gold,traces,questions=questions,diagnostics=diagnostics,source_ids=['s','extra'])
        self.assertEqual(result['Q'],1)
        broken=copy.deepcopy(answers);broken[0]['citations']=['extra']
        result=evaluate(broken,gold,traces,questions=questions,diagnostics=diagnostics,source_ids=['s','extra'])
        self.assertEqual(result['details'][0]['score'],0)
        allowed,errors=final_context(diagnostics[0],['s','extra'])
        self.assertEqual(allowed,['s']);self.assertEqual(errors,[])

    def test_canonical_records_are_not_normalized_or_deduplicated(self):
        q={'id':'q','options':[{'id':'A'}]}
        base={'id':'q','answer_ids':['A'],'citations':['s'],'abstain':False,'rationale':'Proof'}
        self.assertEqual(answer_errors(base,q,['s']),[])
        for patch in [{'answer_ids':['A','A']},{'citations':['s','s']},{'citations':['[s]']},
                      {'rationale':''},{'rationale':' '*1300+'Proof'},{'abstain':'false'},
                      {'answer_ids':['invented']},{'citations':['invented']},{'citations':['']},
                      {'rationale':False}]:
            with self.subTest(patch=patch):self.assertTrue(answer_errors({**base,**patch},q,['s']))
        self.assertTrue(answer_errors(base,q,['s'],no_context=True))
        self.assertEqual(answer_errors({**base,'citations':[]},q,[],no_context=True),[])

    def test_accounting_requires_all_calls_not_just_true_flag(self):
        t,d=accounting_fixture()
        self.assertEqual(accounting_errors(t,d),[])
        self.assertTrue(accounting_errors(t,{}))
        self.assertTrue(accounting_errors(t,{'calls':d['calls'][:1]}))
        for field,value in [('prompt_tokens',-1),('prompt_tokens',True),('completion_tokens',33)]:
            broken=copy.deepcopy(d);broken['calls'][0]['usage'][field]=value
            self.assertTrue(accounting_errors(t,broken))
        broken=copy.deepcopy(d);broken['calls'][0]['max_tokens']=False
        self.assertTrue(accounting_errors(t,broken))
        broken=copy.deepcopy(d);broken['calls'][0]['usage']['prompt_tokens']=8192
        self.assertTrue(accounting_errors(t,broken))
        self.assertTrue(accounting_errors({**t,'input_tokens':21},d))
        self.assertTrue(accounting_errors({**t,'usage_verified':False},d))
        for field,value in [('llm_calls',True),('input_tokens',-1),('output_tokens',4097),('elapsed_ms',False)]:
            self.assertTrue(trace_errors({**t,field:value}))

    def test_legacy_single_and_explicit_zero_call_failure(self):
        t,d=accounting_fixture(1)
        self.assertEqual(accounting_errors(t,d['calls'][0]),[])
        zero={'id':'q','llm_calls':0,'input_tokens':0,'output_tokens':0,
              'usage_verified':True,'technical_error':True,'elapsed_ms':0}
        self.assertEqual(accounting_errors(zero,{'error':'Prompt exceeds window','calls':[]}),[])
        self.assertTrue(accounting_errors(zero,{}))
        self.assertTrue(accounting_errors({**zero,'output_tokens':1},{'error':'preflight'}))

    def test_evaluator_zeroes_invalid_rows_and_retains_missing_question(self):
        answers,gold,traces,questions=evaluation_fixture()
        for patch in [{'answer_ids':['A','A']},{'citations':['s','s']},{'rationale':''},
                      {'abstain':'false'},{'answer_ids':['X']}]:
            broken=copy.deepcopy(answers);broken[0].update(patch)
            result=evaluate(broken,gold,traces,questions=questions,source_ids=['s'])
            self.assertEqual(result['details'][0]['score'],0)
            self.assertTrue(result['details'][0]['validation_errors'])
        result=evaluate(answers[:-1],gold,traces,questions=questions,source_ids=['s'])
        self.assertEqual(len(result['details']),4)
        self.assertEqual(result['Q'],.75)
        valid_wrong=copy.deepcopy(answers);valid_wrong[0]['answer_ids']=['B']
        r=evaluate(valid_wrong,gold,traces,questions=questions)
        self.assertEqual(r['details'][0]['validation_errors'],[])
        with self.assertRaises(ValueError):evaluate(answers+[answers[0]],gold,traces)

    def test_unverified_or_invalid_cost_is_never_reported_as_measured(self):
        a,g,t,q=evaluation_fixture()
        t[0].pop('calls')
        r=evaluate(a,g,t,questions=q)
        self.assertEqual(r['Q'],1)
        self.assertFalse(r['efficiency_eligible'])
        self.assertIsNone(r['mean_cost'])
        t[0]['input_tokens']=-100
        r=evaluate(a,g,t,questions=q)
        self.assertEqual(r['details'][0]['score'],0)
        self.assertIsNone(r['mean_cost'])

    def test_standalone_evaluator_cli_rejects_review_counterexample(self):
        a,g,t,q=evaluation_fixture()
        for row in a:
            row['rationale']=''
            if not row['abstain']:row['answer_ids']*=2;row['citations']*=2
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,rows in [('answers',a),('gold',g),('trace',t),('questions',q),('corpus',[{'block_id':'s'}])]:
                (root/(name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in rows))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate.py'),
                *[v for name in ['answers','gold','trace','questions','corpus'] for v in ['--'+name,str(root/(name+'.jsonl'))]],
                '--output',str(root/'metrics.json')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads((root/'metrics.json').read_text())['Q'],0)

    def test_validator_cli_rejects_multicall_without_evidence(self):
        t,_=accounting_fixture()
        rows={'questions':[{'id':'q','question':'Q','options':[{'id':'A','text':'A'}]}],
              'corpus':[{'block_id':'s'}],
              'answers':[{'id':'q','answer_ids':[],'citations':[],'abstain':True,'rationale':'Not enough data'}],
              'trace':[t],'trace.diagnostics':[{'id':'q','context_block_ids':[]}]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,data in rows.items():(root/(name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in data))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/validate_run.py'),
                *[v for name in ['questions','corpus','answers','trace'] for v in ['--'+name,str(root/(name+'.jsonl'))]]],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('missing or incomplete evidence',result.stderr)

    def test_validator_checks_citation_against_last_call_context(self):
        trace,accounting=accounting_fixture(1)
        diagnostics=[{'id':'q','context_block_ids':['s'],'context':[{'chunk_id':'c','block_id':'s'}],
            'calls':[{**accounting['calls'][0],'context_block_ids':[],'context_chunk_ids':[]}]}]
        rows={'questions':[{'id':'q','question':'Q','options':[{'id':'A','text':'A'}]}],
              'corpus':[{'block_id':'s'}],
              'answers':[{'id':'q','answer_ids':['A'],'citations':['s'],'abstain':False,'rationale':'Proof'}],
              'trace':[trace],'trace.diagnostics':diagnostics}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,data in rows.items():(root/(name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in data))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/validate_run.py'),
                *[v for name in ['questions','corpus','answers','trace'] for v in ['--'+name,str(root/(name+'.jsonl'))]],
                ],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
