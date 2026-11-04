import re
from collections import Counter
from statistics import median

from .formulas import SYMBOLS, parse_formula


MATH_FONT = re.compile(r'^(CM|MSAM|MSBM|EU|Math|Symbol)', re.I)


def fix_text(text):
    """Исправляет ошибочную кодировку кириллицы из шрифтов учебника."""
    text = re.sub(r'\x01(.)', lambda match: match[1]+'\u0301', text)
    return ''.join(c.encode('latin1').decode('cp1251') if 192 <= ord(c) <= 255 else '-' if c in ('\x15', '\x16') else '§' if c == '\x9f' else c for c in text)


def dehyphenate_prose(text):
    """Убирает перенос слова через дефис с учётом отступа следующей строки."""
    return re.sub(r'([A-Za-zА-Яа-яЁё])-\n\s*([A-Za-zА-Яа-яЁё])', r'\1\2', text)


def is_math(span):
    """Определяет математический шрифт в PDF."""
    if span['font'] in ('CMR9', 'CMTI9'):
        return False
    return bool(MATH_FONT.match(span['font']))


def formula_geometry(spans, drawings):
    """Return deduplicated glyph geometry and drawing rules for formula spans."""
    chars = []
    for span in spans:
        for c in span['chars']:
            if c['c'] in (' ', '\n', '\t'):
                continue
            token = SYMBOLS.get(c['c'], c['c'])
            token = {'µ': r'\mu', 'ζ': r'\zeta'}.get(c['c'], token)
            if span['font'].startswith('CMEX'):
                token = {'P': r'\sum', 'Q': r'\prod', 'R': r'\int', 'n': r'\{', 'o': r'\}', '\x12': r'(', '\x13': r')'}.get(c['c'], token)
            if span['font'].startswith('MSBM') and token.isascii() and token.isalpha():
                token = r'\mathbb{'+token+'}'
            elif span['font'].startswith('CMSY'):
                token = {'\\': r'\setminus', '{': r'\{', '}': r'\}'}.get(c['c'], token)
            chars.append({'text': token, 'x0': c['bbox'][0], 'x1': c['bbox'][2], 'y': c['origin'][1], 'size': span['size'], 'font': span['font']})
    chars = collapse_cmex_delimiters(chars)
    unique = []
    for c in sorted(chars, key=lambda c: (c['x0'], c['y'], c['text'])):
        if not any(c['text'] == old['text'] and abs(c['x0']-old['x0']) < 0.4 and abs(c['y']-old['y']) < 0.4 for old in unique):
            unique.append(c)
    chars = unique
    if not chars:
        return [], []
    x0, x1 = min(c['x0'] for c in chars), max(c['x1'] for c in chars)
    y0 = min(s['bbox'][1] for s in spans)
    y1 = max(s['bbox'][3] for s in spans)
    bars = [b for b in drawings if b[0] >= x0-2 and b[1] <= x1+2 and y0-2 <= b[2] <= y1+2]
    return chars, bars


def extract_formula(spans, drawings):
    """Восстанавливает формулу из символов и линий её области."""
    chars, bars = formula_geometry(spans, drawings)
    if not chars:
        return '', []
    warnings = []
    try:
        largest = max(c['size'] for c in chars)
        def fraction_rule(bar):
            within = [c for c in chars if bar[0]-.5 <= (c['x0']+c['x1'])/2 <= bar[1]+.5]
            return (any(bar[2]-largest*1.4 < c['y'] < bar[2] for c in within)
                    and any(bar[2]+1 < c['y'] < bar[2]+largest*1.4 for c in within))
        # Several aligned estimate lines can share one PDF text block. Fraction
        # rules provide a more stable equation-row anchor than glyph baselines:
        # numerator and denominator glyphs sit on opposite sides of the same rule.
        bar_clusters = []
        for bar in sorted((b for b in bars if fraction_rule(b)), key=lambda b: b[2]):
            if not bar_clusters or abs(bar[2]-bar_clusters[-1][0]) > 2.5:
                bar_clusters.append([bar[2], [bar]])
            else:
                bar_clusters[-1][1].append(bar)
                bar_clusters[-1][0] = sum(item[2] for item in bar_clusters[-1][1]) / len(bar_clusters[-1][1])
        bar_anchors = [cluster[0] for cluster in bar_clusters]
        used_bar_rows = len(bar_anchors) > 1 and min(b-a for a, b in zip(bar_anchors, bar_anchors[1:])) > largest*1.8
        if used_bar_rows:
            row_chars = [[] for _ in bar_anchors]
            for c in chars:
                nearest = min(range(len(bar_anchors)), key=lambda i: abs(c['y']-bar_anchors[i]))
                row_chars[nearest].append(c)
            # Keep accents/root rules too: they need not lie on the fraction
            # axis, but belong to the closest equation row.
            row_bars = [[b for b in bars if min(range(len(bar_anchors)),
                         key=lambda j: abs(b[2]-bar_anchors[j])) == i]
                        for i in range(len(bar_anchors))]
            tex = r'\begin{gathered}' + r'\\'.join(
                parse_formula(row, row_bars[i], version=6) for i, row in enumerate(row_chars) if row
            ) + r'\end{gathered}'
        if not used_bar_rows:
            regular = [c for c in chars if c['size'] >= largest*0.9]
            rows = []
            for c in sorted(regular, key=lambda c: c['y']):
                if not rows or abs(c['y']-rows[-1][0]['y']) > 2.5:
                    rows.append([c])
                else:
                    rows[-1].append(c)
            rows = [row for row in rows if len(row) >= 8]
            # Numerator/denominator baselines belong to one expression. Their
            # vertical gap must not trigger the multiline-equation heuristic.
            has_fraction = (any(fraction_rule(b) for b in bars)
                            and max(c['y'] for c in regular)-min(c['y'] for c in regular) <= largest*2.8)
            if not has_fraction and len(rows) > 1 and max(row[0]['y'] for row in rows)-min(row[0]['y'] for row in rows) > largest*0.9:
                baselines = [sum(c['y'] for c in row)/len(row) for row in rows]
                row_chars = [list(row) for row in rows]
                for c in chars:
                    if c not in regular:
                        i = min(range(len(baselines)), key=lambda i: abs(c['y']-baselines[i]))
                        if abs(c['y']-baselines[i]) < largest*1.3:
                            row_chars[i].append(c)
                # Keep distinct baselines as distinct display-math rows. Joining them
                # with a math-space command flattened multi-line derivations into one
                # expression and interleaved unrelated terms on dense textbook pages.
                tex = r'\begin{gathered}' + r'\\'.join(
                    parse_formula(row, bars, version=6) for row in row_chars
                ) + r'\end{gathered}'
            else:
                tex = parse_formula(chars, bars, version=6)
    except (ValueError, RecursionError, IndexError) as error:
        tex = ''.join(c['text'] for c in chars)
        warnings.append(type(error).__name__)
    if any(ord(c) < 32 for c in tex):
        warnings.append('control_characters')
    if '\ufffd' in tex or any(0xe000 <= ord(c) <= 0xf8ff for c in tex):
        warnings.append('unknown_glyphs')
    if re.search(r'([_^])\{[^{}]*\}[_^]\{[^{}]*\}\1', tex):
        warnings.append('repeated_scripts')
    if tex.startswith(('^', '_')):
        warnings.append('leading_script')
    return tex, warnings


def collapse_cmex_delimiters(chars):
    """Сворачивает составные CMEX-глифы больших фигурных скобок в LaTeX-скобки."""
    families = (
        ({'\uf8eb', '\uf8ed'}, {'\uf8f6', '\uf8f8'}, {'\uf8ec', '\uf8f7'}, '(', ')'),
        ({'\uf8ee', '\uf8f0'}, {'\uf8f9', '\uf8fb'}, {'\uf8ef', '\uf8fa'}, '[', ']'),
        ({'\uf8f1', '\uf8f2', '\uf8f3'}, {'\uf8fc', '\uf8fd', '\uf8fe'}, {'\uf8f4'}, r'\{', r'\}'),
    )
    known_parts = set().union(*(left | right | ext for left, right, ext, _, _ in families))
    candidates = [c for c in chars if c.get('font', '').startswith('CMEX') and c['text'] in known_parts]
    groups = []
    for c in sorted(candidates, key=lambda item: item['x0']):
        group = next((g for g in groups if abs(g[0]['x0']-c['x0']) <= 0.75), None)
        if group is None:
            groups.append([c])
        else:
            group.append(c)

    replacements = []
    removed = set()
    for group in groups:
        texts = {c['text'] for c in group}
        delimiter = next((left_tex if texts & left else right_tex
                          for left, right, ext, left_tex, right_tex in families
                          if bool(texts & left) != bool(texts & right)), None)
        if delimiter is None:
            continue
        replacements.append({
            'text': delimiter,
            'x0': min(c['x0'] for c in group),
            'x1': max(c['x1'] for c in group),
            'y': sum(c['y'] for c in group) / len(group),
            'size': max(c['size'] for c in group),
            'font': 'CMEX10',
            'extensible': True,
        })
        removed.update(id(c) for c in group)
    return [c for c in chars if id(c) not in removed] + replacements


def page_stats(page):
    """Считает математические символы и группы строк для ранжирования."""
    math_chars, total, math_lines = 0, 0, 0
    fonts = Counter()
    for b in page.get_text('rawdict')['blocks']:
        for line in b.get('lines', []):
            has_math = False
            for span in line['spans']:
                n = sum(not c['c'].isspace() for c in span['chars'])
                total += n
                fonts[span['font']] += n
                if is_math(span):
                    math_chars += n
                    has_math |= n > 0
            math_lines += has_math
    return {'page': page.number+1, 'math_chars': math_chars, 'total_chars': total, 'math_lines': math_lines, 'fonts': dict(fonts)}


def line_baseline(line):
    """Оценивает базовую линию по обычным символам, игнорируя высокие скобки."""
    ys = [c['origin'][1] for s in line['spans'] if s['size'] >= 8.5 and not s['font'].startswith('CMEX') for c in s['chars']]
    return median(ys) if ys else None


def attach_inline_math_blocks(blocks):
    """Присоединяет отдельные математические строки к соседней строке с прозой."""
    prose = []
    math = []
    for block in blocks:
        spans = [s for line in block.get('lines', []) for s in line['spans']]
        has_words = any(not is_math(s) and any(c['c'].isalpha() for c in s['chars']) for s in spans)
        if has_words:
            prose.append(block)
        elif spans and any(is_math(s) for s in spans):
            math.append(block)

    attached = set()
    for block in math:
        remaining = []
        for line in block['lines']:
            baseline = line_baseline(line)
            if baseline is None:
                # Tall CMEX operators may share an inline math line even though
                # their glyph baseline sits above the ordinary math baseline.
                source_y0, source_y1 = line['bbox'][1], line['bbox'][3]
                glyph_only = bool(line['spans']) and all(is_math(s) for s in line['spans'])
                if not glyph_only:
                    remaining.append(line)
                    continue
            x0, x1 = line['bbox'][0], line['bbox'][2]
            targets = []
            for text_block in prose:
                for target_line in text_block['lines']:
                    target_baseline = line_baseline(target_line)
                    if target_baseline is None:
                        continue
                    if baseline is not None:
                        if abs(target_baseline-baseline) > 2.5:
                            continue
                        target_spans = [target_line]
                        vertical_distance = abs(target_baseline-baseline)
                        max_xgap = 20
                    else:
                        if not source_y0-1 <= target_baseline <= source_y1+1:
                            continue
                        target_spans = [s for s in target_line['spans'] if is_math(s)]
                        vertical_distance = abs(target_baseline-(source_y0+source_y1)/2)
                        max_xgap = 12
                    for target_span in target_spans:
                        tx0, tx1 = target_span['bbox'][0], target_span['bbox'][2]
                        xgap = max(tx0-x1, x0-tx1, 0)
                        if xgap <= max_xgap:
                            targets.append((xgap, vertical_distance, target_line))
            if not targets:
                remaining.append(line)
                continue
            target = min(targets, key=lambda item: (item[0], item[1]))[2]
            target['spans'].extend(line['spans'])
            target['spans'].sort(key=lambda span: span['bbox'][0])
            target['bbox'] = (
                min(target['bbox'][0], line['bbox'][0]),
                min(target['bbox'][1], line['bbox'][1]),
                max(target['bbox'][2], line['bbox'][2]),
                max(target['bbox'][3], line['bbox'][3]),
            )
            attached.add(id(line))
        block['lines'] = [line for line in block['lines'] if id(line) not in attached]

    return [block for block in blocks if block['lines']]


def merge_inline_pdf_lines(blocks):
    """Объединяет перекрывающиеся блоки PDF, набранные на одной строке."""
    changed = True
    while changed:
        changed = False
        for target_block in blocks:
            for target_line in target_block['lines']:
                baseline = line_baseline(target_line)
                if baseline is None:
                    continue
                target_math = any(is_math(span) for span in target_line['spans'])
                for source_block in blocks:
                    if source_block is target_block:
                        continue
                    for source_line in source_block['lines']:
                        source_baseline = line_baseline(source_line)
                        if source_baseline is None or abs(source_baseline-baseline) > 2.5:
                            continue
                        if not target_math and not any(is_math(span) for span in source_line['spans']):
                            continue
                        xgap = max(target_line['bbox'][0]-source_line['bbox'][2],
                                   source_line['bbox'][0]-target_line['bbox'][2], 0)
                        if xgap > 20:
                            continue
                        target_line['spans'].extend(source_line['spans'])
                        target_line['spans'].sort(key=lambda span: span['bbox'][0])
                        target_line['bbox'] = (
                            min(target_line['bbox'][0], source_line['bbox'][0]),
                            min(target_line['bbox'][1], source_line['bbox'][1]),
                            max(target_line['bbox'][2], source_line['bbox'][2]),
                            max(target_line['bbox'][3], source_line['bbox'][3]),
                        )
                        target_block['bbox'] = (
                            min(target_block['bbox'][0], source_line['bbox'][0]),
                            min(target_block['bbox'][1], source_line['bbox'][1]),
                            max(target_block['bbox'][2], source_line['bbox'][2]),
                            max(target_block['bbox'][3], source_line['bbox'][3]),
                        )
                        source_block['lines'].remove(source_line)
                        changed = True
                        break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break
    return [block for block in blocks if block['lines']]


def merge_script_fragments(formulas, text):
    """Склеивает геометрически соседний базовый символ и вынесенный индекс."""
    changed = True
    while changed:
        changed = False
        for script in formulas:
            suffix = script['tex'][:1]
            if suffix not in ('_', '^'):
                continue
            if script['kind'] == 'display' and suffix != '_':
                continue
            if script['kind'] not in ('inline', 'display'):
                continue
            for base in formulas:
                if base is script or base['kind'] not in ('inline', 'display'):
                    continue
                mixed_kinds = base['kind'] != script['kind']
                if mixed_kinds and (suffix != '_' or base['tex'].strip() not in
                                    (r'\int', r'\oint', r'\sum', r'\prod', r'\max', r'\min')):
                    continue
                if suffix == '_' and re.search(r'_\{[^{}]*\}$', base['tex']):
                    continue
                a, b = base['bbox'], script['bbox']
                xgap = max(a[0]-b[2], b[0]-a[2], 0)
                yoverlap = min(a[3], b[3])-max(a[1], b[1])
                if xgap > 12 or yoverlap <= 0:
                    continue
                base_wrapper = '$$' if base['kind'] == 'display' else '$'
                script_wrapper = '$$' if script['kind'] == 'display' else '$'
                pattern = re.escape(base_wrapper+base['tex']+base_wrapper) + r'\s+' + re.escape(script_wrapper+script['tex']+script_wrapper)
                merged_tex = (base['tex'].rstrip() if mixed_kinds else base['tex'])+' '+script['tex']
                merged_wrapper = '$$' if mixed_kinds or base['kind'] == 'display' else '$'
                replacement = merged_wrapper+merged_tex+merged_wrapper
                updated_text, count = re.subn(pattern, lambda _: replacement, text, count=1)
                if not count:
                    continue
                text = updated_text
                base['tex'] = merged_tex
                base['kind'] = 'display' if merged_wrapper == '$$' else 'inline'
                base['bbox'] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                base['warnings'] = [warning for warning in base['warnings']+script['warnings'] if warning != 'leading_script']
                formulas.remove(script)
                changed = True
                break
            if changed:
                break
    return formulas, text


def merge_fraction_fragments(formulas, text, drawings):
    """Повторно разбирает числитель и знаменатель, разнесённые по PDF-блокам."""
    for x0, x1, y in drawings:
        sides = []
        for formula in formulas:
            chars = [char for span in formula.get('_spans', []) for char in span['chars']
                     if x0-0.5 <= (char['bbox'][0]+char['bbox'][2])/2 <= x1+0.5]
            above = any(char['origin'][1] < y-0.5 for char in chars)
            below = any(char['origin'][1] > y+0.5 for char in chars)
            if above != below:
                sides.append((formula, 'above' if above else 'below'))
        for numerator, numerator_side in sides:
            if numerator_side != 'above':
                continue
            for denominator, denominator_side in sides:
                if denominator_side != 'below' or numerator is denominator:
                    continue
                a, b = numerator['bbox'], denominator['bbox']
                if max(a[0]-b[2], b[0]-a[2], 0) > 12 or min(a[3], b[3])-max(a[1], b[1]) <= 0:
                    continue
                spans = numerator.get('_spans', [])+denominator.get('_spans', [])
                if not spans:
                    continue
                numerator_wrapper = '$$' if numerator['kind'] == 'display' else '$'
                denominator_wrapper = '$$' if denominator['kind'] == 'display' else '$'
                wrapper = '$$' if '$$' in (numerator_wrapper, denominator_wrapper) else '$'
                replacement_tex, warnings = extract_formula(spans, drawings)
                replacement = wrapper+replacement_tex+wrapper
                after_numerator = re.escape(numerator_wrapper+numerator['tex']+numerator_wrapper) + r'\s+' + re.escape(denominator_wrapper+denominator['tex']+denominator_wrapper)
                updated_text, count = re.subn(after_numerator, lambda _: replacement, text, count=1)
                if not count:
                    reverse = re.escape(denominator_wrapper+denominator['tex']+denominator_wrapper) + r'\s+' + re.escape(numerator_wrapper+numerator['tex']+numerator_wrapper)
                    updated_text, count = re.subn(reverse, lambda _: replacement, text, count=1)
                if not count:
                    continue
                text = updated_text
                merged = {
                    'tex': replacement_tex,
                    'bbox': [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])],
                    'kind': 'display' if wrapper == '$$' else 'inline',
                    'warnings': [warning for warning in warnings if warning != 'leading_script'],
                    '_spans': spans,
                }
                formulas[formulas.index(numerator)] = merged
                formulas.remove(denominator)
                break
            else:
                continue
            break
    return formulas, text


def extract_page(page):
    """Извлекает абзацы, встроенные и отдельно стоящие формулы."""
    drawings = []
    for d in page.get_drawings():
        for item in d['items']:
            if item[0] == 'l' and abs(item[1].y-item[2].y) < 0.5 and abs(item[1].x-item[2].x) > 2:
                drawings.append((min(item[1].x,item[2].x), max(item[1].x,item[2].x), item[1].y))
    paragraphs, formulas = [], []
    prose_before = []
    blocks = merge_inline_pdf_lines(attach_inline_math_blocks(merge_math_blocks(page.get_text('rawdict', sort=True)['blocks'])))

    for block in blocks:
        lines = block.get('lines', [])
        spans = [s for line in lines for s in line['spans']]
        if not spans:
            continue
        has_prose = any(not is_math(s) and any(c['c'].isalpha() for c in s['chars']) for s in spans)
        if not has_prose and any(is_math(s) for s in spans):
            tex, warnings = extract_formula(spans, drawings)
            formulas.append({'tex': tex, 'bbox': list(block['bbox']), 'kind': 'display', 'warnings': warnings, '_spans': spans})
            paragraphs.append(('$$'+tex+'$$', block['bbox']))
            continue
        output = []
        for line in lines:
            pending = []
            for span in line['spans']+[None]:
                if span is not None and is_math(span):
                    pending.append(span)
                    continue
                if pending:
                    tex, warnings = extract_formula(pending, drawings)
                    formulas.append({'tex': tex, 'kind': 'inline', 'warnings': warnings, 'bbox': [min(s['bbox'][0] for s in pending), min(s['bbox'][1] for s in pending), max(s['bbox'][2] for s in pending), max(s['bbox'][3] for s in pending)], '_spans': list(pending)})
                    output.append('$'+tex+'$')
                    pending = []
                if span is not None:
                    original = ''.join(c['c'] for c in span['chars'])
                    text = fix_text(original)
                    prose_before.append(text)
                    output.append(text)
            output.append('\n')
        paragraphs.append((''.join(output).strip(), block['bbox']))
    paragraph_text = []
    previous_bbox = None
    for paragraph, bbox in paragraphs:
        if not paragraph:
            continue
        same_line = previous_bbox is not None and min(previous_bbox[3], bbox[3]) > max(previous_bbox[1], bbox[1]) and max(0, previous_bbox[0]-bbox[2], bbox[0]-previous_bbox[2]) <= 20
        if paragraph_text and same_line:
            paragraph_text.append(' ')
        elif paragraph_text:
            paragraph_text.append('\n\n')
        paragraph_text.append(paragraph)
        previous_bbox = bbox
    text = ''.join(paragraph_text)
    formulas, text = merge_fraction_fragments(formulas, text, drawings)
    formulas, text = merge_script_fragments(formulas, text)
    text = dehyphenate_prose(text)
    text = re.sub(r'([а-яё])\$', r'\1 $', text)
    text = re.sub(r'\$([а-яё])', r'$ \1', text)
    for formula in formulas:
        formula.pop('_spans', None)
    return {'page': page.number+1, 'text': text, 'formulas': formulas,
            'prose_char_count': sum(len(t) for t in prose_before)}


def merge_math_blocks(blocks):
    """Объединяет соседние математические блоки одной области страницы."""
    prose, groups = [], []
    for block in blocks:
        spans = [s for line in block.get('lines', []) for s in line['spans']]
        pure = spans and any(is_math(s) for s in spans) and not any(not is_math(s) and any(c['c'].isalpha() for c in s['chars']) for s in spans)
        if pure:
            groups.append(dict(block))
        else:
            prose.append(block)
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(groups):
            for j in range(i+1, len(groups)):
                b = groups[j]
                ar, br = a['bbox'], b['bbox']
                xgap = max(ar[0]-br[2], br[0]-ar[2], 0)
                ygap = max(ar[1]-br[3], br[1]-ar[3], 0)
                has_cmex_part = lambda block: any(
                    s['font'].startswith('CMEX') and any(c['c'] in ('\uf8eb', '\uf8ed', '\uf8ec', '\uf8f6', '\uf8f8', '\uf8f7', '\uf8ee', '\uf8f0', '\uf8ef', '\uf8f9', '\uf8fb', '\uf8fa', '\uf8f1', '\uf8f2', '\uf8f3', '\uf8fc', '\uf8fd', '\uf8fe', '\uf8f4') for c in s['chars'])
                    for line in block.get('lines', []) for s in line['spans']
                )
                paired_vertical_piece = (
                    has_cmex_part(a) and has_cmex_part(b)
                    and abs(ar[0]-br[0]) <= 1 and xgap <= 1 and ygap <= 12
                )
                def operator_limits(operator_block, limit_block):
                    limit_spans = [s for line in limit_block.get('lines', []) for s in line['spans']]
                    limit_chars = [c for span in limit_spans for c in span['chars'] if not c['c'].isspace()]
                    if not limit_chars or any(not is_math(span) for span in limit_spans):
                        return False
                    if max(span['size'] for span in limit_spans) >= 8.5:
                        return False
                    for line in operator_block.get('lines', []):
                        for span in line['spans']:
                            if not span['font'].startswith('CMEX'):
                                continue
                            for char in span['chars']:
                                if char['c'] not in ('Z', 'I', 'X', 'Y', 'Q', 'P', 'R'):
                                    continue
                                cb = char['bbox']
                                center = (cb[0]+cb[2])/2
                                limit_center = (limit_block['bbox'][0]+limit_block['bbox'][2])/2
                                if abs(center-limit_center) > max(5, span['size']*0.65):
                                    continue
                                if span['size'] < max(s['size'] for s in limit_spans)*1.25:
                                    continue
                                if ygap <= span['size']*1.1:
                                    return True
                    return False

                operator_limit = operator_limits(a, b) or operator_limits(b, a)
                if paired_vertical_piece or operator_limit or (xgap <= 20 and ygap <= 3):
                    a['bbox'] = (min(ar[0],br[0]), min(ar[1],br[1]), max(ar[2],br[2]), max(ar[3],br[3]))
                    a['lines'] = a['lines']+b['lines']
                    groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return sorted(prose+groups, key=lambda b: (b['bbox'][1], b['bbox'][0]))


def to_latex_document(text):
    """Создаёт LaTeX-документ с русским текстом и математическими вставками."""
    escape = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    parts = re.split(r'(\$\$.*?\$\$|\$[^$]*?\$)', text, flags=re.S)
    body = ''.join(part if part.startswith('$') else ''.join(escape.get(c,c) for c in part) for part in parts)
    return r'\documentclass[12pt]{article}\usepackage{fontspec}\setmainfont{Liberation Serif}\usepackage{amsmath,amssymb}\begin{document}'+'\n'+body+'\n'+r'\end{document}'+'\n'
