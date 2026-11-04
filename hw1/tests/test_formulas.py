import sys
import json
import copy
import re
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.formulas import parse_formula
from src.extraction import attach_inline_math_blocks, dehyphenate_prose, extract_formula, fix_text, is_math, merge_inline_pdf_lines, merge_math_blocks, merge_script_fragments
from src.formula_layout import serialize


def symbol(text, x, y, size=12):
    """Создаёт геометрическое описание символа для проверки правил."""
    return {'text': text, 'x0': x, 'x1': x+5, 'y': y, 'size': size}


class FormulaTests(unittest.TestCase):
    def test_textbook_page36_cauchy_kernel_series_geometry(self):
        from scripts.check_formulas import normalize
        fixture=json.loads((Path(__file__).parent/'textbook_formula_p036.json').read_text())
        actual=parse_formula(fixture['chars'],[tuple(b) for b in fixture['bars']],version=6)
        self.assertEqual(normalize(actual),normalize(fixture['expected']))

    def test_nested_fractions_and_roots_real_geometry_with_variants(self):
        from scripts.check_formulas import normalize
        fixtures=json.loads((Path(__file__).parent/'nested_formula_geometry.json').read_text())
        for fixture in fixtures:
            for scale in (0.7,1.,2.):
                for neighbor in (False,True):
                    with self.subTest(case=fixture['id'],scale=scale,neighbor=neighbor):
                        chars=copy.deepcopy(fixture['chars'])
                        replacements={'x':'t'} if fixture['id']=='frac_02' else {'a':'u','b':'v','c':'w','d':'t'}
                        expected=re.sub(r'\b([xabcd])\b',lambda m:replacements.get(m[1],m[1]),fixture['expected'])
                        for c in chars:
                            c['text']=replacements.get(c['text'],c['text'])
                            for key in ('x0','x1'):c[key]=c[key]*scale+17
                            c['y']=c['y']*scale+30
                            c['size']*=scale
                        bars=[(x*scale+17,xx*scale+17,y*scale+30) for x,xx,y in fixture['bars']]
                        if neighbor:
                            bar=max(bars,key=lambda b:b[1]-b[0]);size=max(c['size'] for c in chars)
                            x=max(c['x1'] for c in chars)+size
                            chars.extend([{'text':s,'x0':x+i*size,'x1':x+(i+.5)*size,
                                           'y':bar[2]+size*.25,'size':size} for i,s in enumerate(['+','k'])])
                            expected+='+k'
                        self.assertEqual(normalize(parse_formula(chars,bars,version=6)),normalize(expected))

    def test_real_pdf_fraction_chains_and_wirtinger_derivation(self):
        fixtures = json.loads((Path(__file__).parent/'real_fraction_fixtures.json').read_text())
        for fixture in fixtures:
            with self.subTest(page=fixture['page'], label=fixture['label']):
                tex, warnings = extract_formula(fixture['spans'], fixture['bars'])
                self.assertEqual(tex.replace(' ', ''), fixture['expected'].replace(' ', ''))
                self.assertEqual(warnings, [])

    def test_pdf_dash_controls(self):
        """Не оставляет управляющие дефисы из кодировки PDF в прозе."""
        self.assertEqual(fix_text('Миттаг\x15Леффлера\x16гамма'), 'Миттаг-Леффлера-гамма')
        self.assertEqual(fix_text('\x9f 1.'), '§ 1.')

    def test_dehyphenate_indented_line_wrap(self):
        """Склеивает слово, перенесённое через дефис с отступом новой строки."""
        self.assertEqual(dehyphenate_prose('ком-\n  плексного функ-\nций analy-\nsis'), 'комплексного функций analysis')

    def test_nine_point_computer_modern_is_prose(self):
        """Не считает библиографические строки шрифтов CMR9/CMTI9 математикой."""
        self.assertFalse(is_math({'font': 'CMR9'}))
        self.assertFalse(is_math({'font': 'CMTI9'}))
        self.assertTrue(is_math({'font': 'CMR10'}))

    def test_unicode_math_symbols(self):
        """Преобразует использованные в учебнике математические Unicode-знаки в LaTeX."""
        span = {'font': 'CMR10', 'size': 10, 'bbox': [0, 0, 30, 10], 'chars': [
            {'c': char, 'bbox': [i*10, 0, i*10+8, 10], 'origin': [i*10, 10]}
            for i, char in enumerate(('△', '↗', '⊆'))
        ]}
        tex, _ = extract_formula([span], [])
        self.assertEqual(tex, r'\triangle \nearrow \subseteq ')

    def test_merge_geometric_script_fragment(self):
        """Присоединяет отдельный индекс к соседней inline-формуле."""
        formulas = [
            {'tex': 'U^{+}', 'kind': 'inline', 'bbox': [0, 0, 15, 10], 'warnings': []},
            {'tex': '_{n}', 'kind': 'inline', 'bbox': [12, 4, 23, 12], 'warnings': ['leading_script']},
        ]
        merged, text = merge_script_fragments(formulas, '$U^{+}$ $_{n}$')
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['tex'], 'U^{+} _{n}')
        self.assertEqual(text, '$U^{+} _{n}$')
        self.assertNotIn('leading_script', merged[0]['warnings'])

    def test_merge_script_using_text_order(self):
        """Сверяет порядок PDF-блоков с порядком формул в извлечённом тексте."""
        formulas = [
            {'tex': '_{n=0}a_{n}z^{n}', 'kind': 'inline', 'bbox': [188, 370, 226, 384], 'warnings': ['leading_script']},
            {'tex': r'\sum \infty ', 'kind': 'inline', 'bbox': [173, 368, 196, 388], 'warnings': []},
        ]
        text = 'Для ряда $\\sum \\infty $ $_{n=0}a_{n}z^{n}$ сходится.'
        merged, updated = merge_script_fragments(formulas, text)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['tex'], r'\sum \infty  _{n=0}a_{n}z^{n}')
        self.assertEqual(updated, 'Для ряда $\\sum \\infty  _{n=0}a_{n}z^{n}$ сходится.')

    def test_merge_inline_operator_with_display_script(self):
        """Объединяет знак интеграла и его нижний предел из разных типов блока."""
        formulas = [
            {'tex': r'\int ', 'kind': 'inline', 'bbox': [10, 0, 15, 10], 'warnings': []},
            {'tex': r'_{\gamma }f(z)=0', 'kind': 'display', 'bbox': [10, 5, 30, 20], 'warnings': ['leading_script']},
        ]
        merged, text = merge_script_fragments(formulas, '$\\int $ $$_{\\gamma }f(z)=0$$')
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['tex'], r'\int _{\gamma }f(z)=0')
        self.assertEqual(merged[0]['kind'], 'display')
        self.assertEqual(text, '$$\\int _{\\gamma }f(z)=0$$')

    def test_merge_pdf_blocks_sharing_math_baseline(self):
        """Склеивает PDF-блоки с общей базовой линией формулы и прозы."""
        def span(font, x, y, char):
            return {'font': font, 'size': 10, 'bbox': [x, y-5, x+5, y+5],
                    'chars': [{'c': char, 'bbox': [x, y-5, x+5, y+5], 'origin': [x, y]}]}

        left = {'bbox': [0, 5, 20, 15], 'lines': [
            {'bbox': [0, 5, 20, 15], 'spans': [span('CMEX10', 10, 10, 'P'), span('CMMI7', 15, 10, 'n')]}]}
        right = {'bbox': [0, 5, 40, 15], 'lines': [
            {'bbox': [0, 5, 40, 15], 'spans': [span('F41', 0, 10, 'word')]}]}
        blocks = merge_inline_pdf_lines([left, right])
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]['lines'][0]['spans']), 3)
        self.assertEqual(blocks[0]['lines'][0]['spans'][0]['font'], 'F41')

    def test_ms_bold_maps_only_latin_letters_to_blackboard(self):
        """Сохраняет греческие символы из MSBM как греческие команды."""
        def span(char, x):
            return {'font': 'MSBM10', 'size': 10, 'bbox': [x, 0, x+5, 10],
                    'chars': [{'c': char, 'bbox': [x, 0, x+5, 10], 'origin': [x, 10]}]}

        greek, _ = extract_formula([span('κ', 0)], [])
        latin, _ = extract_formula([span('T', 10)], [])
        self.assertEqual(greek, r'\kappa ')
        self.assertEqual(latin, r'\mathbb{T}')

    def test_prime_after_closing_delimiter(self):
        """Оставляет производную после закрывающей скобки, не внутри индекса."""
        chars = [
            symbol('a', 0, 20),
            {'text': r')^{\prime}', 'x0': 10, 'x1': 16, 'y': 15,
             'size': 12, 'font': 'CMEX10'},
        ]
        self.assertEqual(serialize(chars, lambda cs, _: ''.join(c['text'] for c in cs)), r'a)^{\prime}')

    def test_cmex_braces(self):
        """Декодирует фигурные и круглые скобки из шрифта CMEX."""
        def cmex(char):
            return {'font': 'CMEX10', 'size': 10, 'bbox': [0, 0, 6, 10],
                    'chars': [{'c': char, 'bbox': [0, 0, 6, 10], 'origin': [0, 10]}]}

        left, _ = extract_formula([cmex('n')], [])
        right, _ = extract_formula([cmex('o')], [])
        summation, _ = extract_formula([cmex('P')], [])
        product, _ = extract_formula([cmex('Q')], [])
        integral, _ = extract_formula([cmex('R')], [])
        self.assertEqual((left, right), (r'\{', r'\}'))
        self.assertEqual(summation, r'\sum ')
        self.assertEqual(product, r'\prod ')
        self.assertEqual(integral, r'\int ')

    def test_cmex_product_limits(self):
        """Декодирует глиф CMEX Q как произведение и присоединяет его пределы."""
        def span(font, char, size, x, y):
            return {'font': font, 'size': size, 'bbox': [x, y-5, x+5, y+5],
                    'chars': [{'c': char, 'bbox': [x, y-5, x+5, y+5], 'origin': [x, y]}]}

        spans = [span('CMEX10', 'Q', 10, 20, 20), span('CMMI7', 'k', 7, 10, 29),
                 span('CMR7', '=', 7, 14, 29), span('CMR7', '1', 7, 18, 29),
                 span('CMMI7', 'n', 7, 21, 11), span('CMMI10', 'a', 10, 29, 20)]
        self.assertEqual(extract_formula(spans, []), (r'\prod_{k=1}^{n}a', []))

    def test_merge_detached_product_limit_block(self):
        """Объединяет отдельный нижний предел с крупным CMEX-оператором."""
        def span(font, char, size, x, y):
            return {'font': font, 'size': size, 'bbox': [x, y-5, x+5, y],
                    'chars': [{'c': char, 'bbox': [x, y-5, x+5, y], 'origin': [x, y]}]}

        operator = span('CMEX10', 'Y', 10, 10, 20)
        body = span('CMMI10', 'a', 10, 22, 20)
        lower = [span('CMMI7', 'k', 7, 8, 30), span('CMR7', '=', 7, 13, 30),
                 span('CMR7', '1', 7, 18, 30)]
        blocks = [
            {'bbox': [10, 10, 27, 20], 'lines': [{'bbox': [10, 10, 27, 20], 'spans': [operator, body]}]},
            {'bbox': [8, 25, 23, 30], 'lines': [{'bbox': [8, 25, 23, 30], 'spans': lower}]},
        ]
        merged = merge_math_blocks(blocks)
        self.assertEqual(len(merged), 1)
        spans = [span for line in merged[0]['lines'] for span in line['spans']]
        self.assertEqual(extract_formula(spans, []), (r'\prod_{k=1}a', []))

    def test_attach_tall_cmex_operator_to_inline_math(self):
        """Присоединяет высокий глиф CMEX к соседнему math-фрагменту по геометрии."""
        operator = {'font': 'CMEX10', 'size': 10, 'bbox': [313, 194, 319, 205],
                    'chars': [{'c': 'R', 'bbox': [313, 194, 319, 205], 'origin': [313, 195]}]}
        math_span = {'font': 'CMMI10', 'size': 10, 'bbox': [325, 197, 331, 210],
                     'chars': [{'c': 'f', 'bbox': [325, 197, 331, 210], 'origin': [325, 203]}]}
        prose_span = {'font': 'F41', 'size': 10, 'bbox': [335, 197, 350, 210],
                      'chars': [{'c': 'word', 'bbox': [335, 197, 350, 210], 'origin': [335, 203]}]}
        text_block = {'bbox': [300, 194, 350, 210], 'lines': [
            {'bbox': [300, 194, 350, 210], 'spans': [math_span, prose_span]}]}
        operator_block = {'bbox': [313, 194, 319, 205], 'lines': [
            {'bbox': [313, 194, 319, 205], 'spans': [operator]}]}
        blocks = attach_inline_math_blocks([text_block, operator_block])
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]['lines'][0]['spans']), 3)

    def test_cmex_extensible_brace_parts(self):
        """Сворачивает части составных CMEX-скобок в обычные LaTeX-скобки."""
        pieces = [
            ('\uf8eb', 0, 10, 'CMEX10', 30), ('\uf8ec', 0, 20, 'CMEX10', 30), ('\uf8ed', 0, 30, 'CMEX10', 30),
            ('\uf8ee', 5, 10, 'CMEX10', 30), ('\uf8ef', 5, 20, 'CMEX10', 30), ('\uf8f0', 5, 30, 'CMEX10', 30),
            ('\uf8f1', 10, 10, 'CMEX10', 30), ('\uf8f4', 10, 20, 'CMEX10', 30), ('\uf8f3', 10, 30, 'CMEX10', 30),
            ('x', 15, 20, 'CMR10', 10),
            ('\uf8fc', 20, 10, 'CMEX10', 30), ('\uf8f4', 20, 20, 'CMEX10', 30), ('\uf8fe', 20, 30, 'CMEX10', 30),
            ('\uf8f9', 25, 10, 'CMEX10', 30), ('\uf8fa', 25, 20, 'CMEX10', 30), ('\uf8fb', 25, 30, 'CMEX10', 30),
            ('\uf8f6', 30, 10, 'CMEX10', 30), ('\uf8f7', 30, 20, 'CMEX10', 30), ('\uf8f8', 30, 30, 'CMEX10', 30),
        ]
        spans = [
            {'font': font, 'size': size, 'bbox': [x-1, y-2, x+5, y+2],
             'chars': [{'c': char, 'bbox': [x-1, y-2, x+5, y+2], 'origin': [x, y]}]}
            for char, x, y, font, size in pieces
        ]
        tex, warnings = extract_formula(spans, [])
        self.assertEqual(tex, r'([\{x\}])')
        self.assertEqual(warnings, [])

    def test_fraction_rows_anchor_on_rule_height(self):
        """Не разрывает числитель и знаменатель в многострочной оценке."""
        def span(char, y):
            return {'font': 'CMR10', 'size': 10, 'bbox': [0, y-5, 10, y+2],
                    'chars': [{'c': char, 'bbox': [2, y-5, 8, y+2], 'origin': [2, y]}]}

        tex, _ = extract_formula([span('a', 8), span('b', 12), span('c', 48), span('d', 52)],
                                 [(0, 10, 10), (0, 10, 50)])
        self.assertIn(r'\frac{a}{b}', tex)
        self.assertIn(r'\frac{c}{d}', tex)
        self.assertIn(r'\\', tex)

    def test_limit_rule_is_not_a_fraction(self):
        """Считает черту под lim оформлением оператора, не дробной чертой."""
        chars = [
            {'text': char, 'x0': x, 'x1': x+4, 'y': y, 'size': size, 'font': 'CMR10'}
            for char, x, y, size in [('l', 0, 10, 10), ('i', 5, 10, 10), ('m', 10, 10, 10),
                                     ('n', -3, 18, 7), (r'\to', 1, 18, 7), (r'\infty', 8, 18, 7)]
        ]
        self.assertEqual(parse_formula(chars, [(0, 15, 13)], version=6), r'\lim_{n\to \infty }')

    def test_left_limit_is_attached_to_lim(self):
        """Присоединяет предел, смещённый влево от знака lim."""
        chars = [
            {'text': 'z', 'x0': 0, 'x1': 5, 'y': 15, 'size': 7, 'font': 'CMMI7'},
            {'text': r'\to', 'x0': 6, 'x1': 13, 'y': 15, 'size': 7, 'font': 'CMSY7'},
            {'text': r'\lim', 'x0': 18, 'x1': 32, 'y': 10, 'size': 10, 'font': 'CMR10'},
            {'text': 'P', 'x0': 34, 'x1': 41, 'y': 10, 'size': 10, 'font': 'CMR10'},
        ]
        self.assertEqual(parse_formula(chars, [], version=6), r'\lim_{z\to}P')

    def test_inf_operator_with_left_limit(self):
        """Собирает inf и его вынесенное влево условие z∈D."""
        chars = [symbol('i', 20, 10), symbol('n', 24, 10), symbol('f', 28, 10),
                 symbol('z', 5, 20, 7), symbol(r'\in', 10, 20, 7), symbol('D', 15, 20, 7),
                 symbol('A', 36, 10)]
        self.assertEqual(parse_formula(chars, [], version=6), r'\inf_{z\in D}A')

    def test_integral_limit_left_of_operator(self):
        """Привязывает вынесенный влево предел интеграла к знаку интеграла."""
        chars = [
            {'text': r'\int', 'x0': 10, 'x1': 16, 'y': 20, 'size': 12, 'font': 'CMEX10'},
            {'text': r'\gamma', 'x0': 5, 'x1': 9, 'y': 30, 'size': 8, 'font': 'CMMI7'},
            {'text': 'f', 'x0': 18, 'x1': 23, 'y': 20, 'size': 12, 'font': 'CMMI10'},
        ]
        self.assertEqual(parse_formula(chars, [], version=6), r'\int_{\gamma }f')

    def test_residue_name_with_left_subscript(self):
        """Собирает res в оператор и присоединяет вынесенный влево индекс."""
        chars = [
            {'text': 'r', 'x0': 10, 'x1': 14, 'y': 20, 'size': 10, 'font': 'CMR10'},
            {'text': 'e', 'x0': 14, 'x1': 18, 'y': 20, 'size': 10, 'font': 'CMR10'},
            {'text': 's', 'x0': 18, 'x1': 22, 'y': 20, 'size': 10, 'font': 'CMR10'},
            {'text': 'z', 'x0': 7, 'x1': 10, 'y': 27, 'size': 7, 'font': 'CMMI7'},
            {'text': '=', 'x0': 11, 'x1': 16, 'y': 27, 'size': 7, 'font': 'CMR7'},
            {'text': 'a', 'x0': 17, 'x1': 20, 'y': 27, 'size': 7, 'font': 'CMMI7'},
            {'text': 'f', 'x0': 24, 'x1': 29, 'y': 20, 'size': 10, 'font': 'CMMI10'},
        ]
        self.assertEqual(parse_formula(chars, [], version=6), r'\operatorname{res}_{z=a}f')

    def test_max_operator_with_left_subscript(self):
        """Собирает вынесенный влево индекс у оператора max."""
        chars = [
            {'text': 'm', 'x0': 10, 'x1': 14, 'y': 20, 'size': 10, 'font': 'CMR10'},
            {'text': 'a', 'x0': 14, 'x1': 18, 'y': 20, 'size': 10, 'font': 'CMR10'},
            {'text': 'x', 'x0': 18, 'x1': 22, 'y': 20, 'size': 10, 'font': 'CMR10'},
            {'text': r'\rho', 'x0': 7, 'x1': 10, 'y': 27, 'size': 7, 'font': 'CMMI7'},
            {'text': '|', 'x0': 11, 'x1': 13, 'y': 27, 'size': 7, 'font': 'CMR7'},
        ]
        self.assertEqual(parse_formula(chars, [], version=6), r'\max_{\rho |}')

    def test_fraction(self):
        """Проверяет восстановление дроби по линии и двум символам."""
        chars = [symbol('a', 10, 10), symbol('b', 10, 25)]
        self.assertEqual(parse_formula(chars, [(8, 18, 15)], 2), r'\frac{a}{b}')

    def test_subscript(self):
        """Проверяет нижний индекс возле базовой строки."""
        chars = [symbol('x', 0, 20), symbol('n', 6, 25, 8), symbol('+', 14, 20), symbol('y', 22, 20)]
        self.assertEqual(parse_formula(chars, [], 2), 'x_{n}+y')

    def test_superscript(self):
        """Проверяет степень возле базовой строки."""
        chars = [symbol('x', 0, 20), symbol('2', 6, 14, 8), symbol('+', 14, 20), symbol('y', 22, 20)]
        self.assertEqual(parse_formula(chars, [], 2), 'x^{2}+y')

    def test_input_unchanged(self):
        """Проверяет повторяемость разбора без изменения входа."""
        chars = [symbol('a', 10, 10), symbol('b', 10, 25)]
        original = [dict(c) for c in chars]
        first = parse_formula(chars, [(8, 18, 15)], 2)
        self.assertEqual(first, parse_formula(chars, [(8, 18, 15)], 2))
        self.assertEqual(chars, original)


if __name__ == '__main__':
    unittest.main()
