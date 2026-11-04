import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate import evaluate, f1


class EvaluationTests(unittest.TestCase):
    def test_content_contradiction_zeroes_score_preserving_automatic_metric(self):
        answers, gold, traces = self.fixture()
        base = evaluate(answers, gold, traces)
        reviewed = evaluate(answers, gold, traces, reviews={
            '0': {'zero_score': True, 'reason': 'Cited example does not support the answer'}})
        self.assertEqual(reviewed['automatic_Q'], base['Q'])
        self.assertEqual(reviewed['details'][0]['score'], 0)
        self.assertLess(reviewed['Q'], base['Q'])
        self.assertEqual(reviewed['answer_accuracy'], base['answer_accuracy'])
        with self.assertRaises(ValueError):
            evaluate(answers, gold, traces, reviews={'unknown': {'zero_score':True, 'reason':'x'}})

    def fixture(self):
        categories = ['условия', 'сопоставление', 'применение', 'недостаточность данных']
        gold, answers, traces = [], [], []
        for i, category in enumerate(categories):
            abstain = i == 3
            base = {'id': str(i), 'abstain': abstain, 'answer_ids': [] if abstain else ['B']}
            gold.append({**base, 'category': category, 'citation_sets': [[]] if abstain else [['s1', 's2']]})
            answers.append({**base, 'citations': [] if abstain else ['s1'], 'rationale':'Supported result'})
            traces.append({'id': str(i), 'llm_calls': 1, 'input_tokens': 100,
                'output_tokens': 10, 'elapsed_ms': 20, 'usage_verified': True,
                'calls':[{'usage':{'prompt_tokens':100,'completion_tokens':10},'max_tokens':32}]})
        return answers, gold, traces

    def test_partial_citations_and_macro(self):
        result = evaluate(*self.fixture())
        self.assertAlmostEqual(result['Q'], (3*(0.8+0.2*2/3)+1)/4)
        self.assertEqual(result['mean_cost'], 396)
        self.assertEqual(f1(['s1'], ['s1', 's2']), 2/3)

    def test_technical_abstention_does_not_gain_refusal_credit(self):
        answers, gold, traces = self.fixture()
        traces[-1]['technical_error'] = True
        result = evaluate(answers, gold, traces)
        self.assertEqual(result['details'][-1]['score'], 0)

    def test_missing_category_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(*[x[:-1] for x in self.fixture()])


if __name__ == '__main__':
    unittest.main()
