import unittest
from src.retrieval import tokenize


class MathSearchTests(unittest.TestCase):
    def test_signed_powers_are_distinct_and_equivalent_encodings_match(self):
        negative=tokenize('z^{-3}')
        for text in ['z^-3','z^{−3}','z⁻³','z^(-3)']:
            self.assertEqual(negative,tokenize(text),text)
        self.assertNotEqual(negative,tokenize('z^{3}'))
        self.assertEqual(tokenize('z^{+3}'),tokenize('z³'))
        self.assertEqual(tokenize('x−y'),tokenize('x-y'))
        self.assertNotEqual(tokenize('x-y'),tokenize('x+y'))

    def test_derivative_order_and_owner_are_preserved(self):
        second=tokenize("f''(0)")
        for text in ['f″(0)','f′′(0)',r'f^{\prime\prime}(0)']:
            self.assertEqual(second,tokenize(text),text)
        self.assertNotEqual(second,tokenize("f'(0)"))
        self.assertNotEqual(second,tokenize('f(0)'))
        self.assertNotEqual(tokenize("f'(g)"),tokenize("f(g')"))
        self.assertNotEqual(tokenize("f'(x)"),tokenize("f(x)'"))
        self.assertNotIn('__math_derivative_y_1',tokenize("Cauchy's theorem"))

    def test_conjugation_modulus_and_factorial_remain_distinct(self):
        self.assertNotEqual(tokenize(r'\bar z'),tokenize('z'))
        self.assertEqual(tokenize(r'\bar{z}'),tokenize(r'\overline{z}'))
        self.assertNotEqual(tokenize('|z|'),tokenize('z'))
        self.assertNotEqual(tokenize('n!'),tokenize('n'))
