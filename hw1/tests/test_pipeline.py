import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm import LLMClient
from src.pipeline import parse_answer


class PipelineTests(unittest.TestCase):
    def test_prefill_restart_and_escaped_value_opening_are_only_syntax_repairs(self):
        q={'id':'test','options':[{'id':'Z'}]}
        original=json.dumps({'answer_ids':['Z'],'abstain':False,'citations':['src'],'rationale':'Text'})
        self.assertEqual(parse_answer('{\n'+original,q,['src'])['answer_ids'],['Z'])
        refused=r'{"answer_ids":[],"abstain":true,"citations":[],"rationale":\"Нет достаточных источников."}'
        self.assertTrue(parse_answer(refused,q,['src'])['abstain'])
        with self.assertRaises(ValueError):
            parse_answer('{\n{\n'+original,q,['src'])

    def test_raw_math_commands_do_not_silently_turn_into_json_controls(self):
        text = r'{"answer_ids":["Z"],"citations":["src"],"abstain":false,"rationale":"$\frac{1}{2}$ and $\bar z \neq 0$"}'
        a = parse_answer(text, {'id':'test','options':[{'id':'Z'}]}, ['src'])
        self.assertEqual(a['rationale'], r'$\frac{1}{2}$ and $\bar z \neq 0$')

    def test_latex_invalid_escape_and_source_wrappers(self):
        text = r'{"answer_ids":["Z"],"citations":["[src]"],"abstain":false,"rationale":"$\mathbb{C}$ \\setminus $\gamma$"}'
        answer = parse_answer(text, {'id': 'test', 'options': [{'id': 'Z'}]}, ['src'])
        self.assertEqual(answer['citations'], ['src'])
        self.assertEqual(answer['rationale'], r'$\mathbb{C}$ \setminus $\gamma$')

    def test_valid_json_escapes_are_preserved(self):
        rationale = 'Line\nTab\tQuote" slash\\ and ' + r'\frac{x}{y}'
        text = json.dumps({'answer_ids': ['Z'], 'citations': ['[literal]'],
                           'abstain': False, 'rationale': rationale})
        answer = parse_answer(text, {'id': 'test', 'options': [{'id': 'Z'}]}, ['[literal]'])
        self.assertEqual(answer['rationale'], rationale)
        self.assertEqual(answer['citations'], ['[literal]'])

    def test_repairs_do_not_allow_unknown_duplicate_or_truncated_data(self):
        q = {'id': 'test', 'options': [{'id': 'Z'}]}
        for citations in [['[unknown]'], ['src', '[src]']]:
            with self.assertRaises(ValueError):
                parse_answer(json.dumps({'answer_ids':['Z'], 'citations':citations,
                             'abstain':False, 'rationale':'Text'}), q, ['src'])
        for text in [r'{"rationale":"\uZZZZ"}', '{"rationale":"unfinished']:
            with self.assertRaises(ValueError):
                parse_answer(text, q, ['src'])

    def test_dangling_markdown_tick_is_removed_but_prose_rejected(self):
        text = json.dumps({'answer_ids': ['Z'], 'citations': ['src'], 'abstain': False, 'rationale': 'Источник'})
        q = {'id': 'test', 'options': [{'id': 'Z'}]}
        self.assertEqual(parse_answer(text+'`', q, ['src'])['answer_ids'], ['Z'])
        with self.assertRaises(ValueError):
            parse_answer(text+' trailing text', q, ['src'])

    def test_batch_encoding_counts_ids_not_fields(self):
        client = object.__new__(LLMClient)
        client.runtime = {'model': {'thinking': True}}
        client.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: {'input_ids': [1]*123, 'attention_mask': [1]*123})
        self.assertEqual(client.count([]), 123)

    def test_invalid_or_ungrounded_answer_rejected(self):
        q = {'id': 'test', 'options': [{'id': 'Z'}]}
        for ids, citations in [(['A'], ['src']), (['Z'], ['invented']), (['Z', 'Z'], ['src'])]:
            with self.assertRaises(ValueError):
                parse_answer(json.dumps({'answer_ids': ids, 'citations': citations,
                    'abstain': False, 'rationale': 'Обоснование'}), q, ['src'])

    def test_boolean_is_not_coerced(self):
        with self.assertRaises(ValueError):
            parse_answer(json.dumps({'answer_ids': [], 'citations': [], 'abstain': 'false',
                'rationale': 'Нет данных'}), {'id': 'test', 'options': []}, [])


if __name__ == '__main__':
    unittest.main()
