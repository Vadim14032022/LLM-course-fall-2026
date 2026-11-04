import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corpus import split_text, validate_corpus


class CorpusTests(unittest.TestCase):
    def test_preserves_multiline_math_and_text(self):
        text = 'Ввод.\n\n$$a\n\nb=c$$\n\nКонец.'
        pieces = split_text(text, 12)
        self.assertEqual(''.join(pieces), text)
        self.assertTrue(any('$$a\n\nb=c$$' in p for p in pieces))

    def test_large_formula_remains_indivisible(self):
        text = '$$' + 'a+' * 100 + 'b$$'
        self.assertEqual(split_text(text, 20), [text])

    def test_accepts_external_ids(self):
        validate_corpus([{'block_id': 'foreign-id:42', 'page': 7, 'section': '', 'text': 'Текст'}])

    def test_rejects_duplicate_ids(self):
        row = {'block_id': 'same', 'page': 1, 'section': '', 'text': 'Текст'}
        with self.assertRaises(ValueError):
            validate_corpus([row, dict(row)])


if __name__ == '__main__':
    unittest.main()
