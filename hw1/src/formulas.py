from collections import Counter
import re

from .formula_layout import extract_array, extract_cases, serialize


SYMBOLS = dict(zip('αβγδεθλμπσφω∂∞∑∏∫∮≤≥≠∈→−', [r'\alpha', r'\beta', r'\gamma', r'\delta', r'\varepsilon', r'\theta', r'\lambda', r'\mu', r'\pi', r'\sigma', r'\varphi', r'\omega', r'\partial', r'\infty', r'\sum', r'\prod', r'\int', r'\oint', r'\leq', r'\geq', r'\neq', r'\in', r'\to', '-']))


SYMBOLS.update({
    '⩽': r'\leq', '⩾': r'\geq', '′': r'\prime', 'τ': r'\tau',
    '△': r'\triangle', '↗': r'\nearrow', '⊆': r'\subseteq',
    '∆': r'\Delta', 'Δ': r'\Delta', 'Γ': r'\Gamma', 'Φ': r'\Phi',
    '∩': r'\cap', '∪': r'\cup', 'ϱ': r'\varrho', 'ρ': r'\rho',
    '□': r'\square', '⊂': r'\subset', '⊃': r'\supset', '≡': r'\equiv',
    '∅': r'\emptyset', 'ν': r'\nu', '±': r'\pm', '∗': r'\ast',
    '∼': r'\sim', '×': r'\times', 'ξ': r'\xi', 'ψ': r'\psi',
    '·': r'\cdot', 'η': r'\eta', 'χ': r'\chi', 'κ': r'\kappa',
    'Θ': r'\Theta', 'Λ': r'\Lambda', 'Π': r'\Pi', 'Σ': r'\Sigma',
    'Ψ': r'\Psi', 'Ω': r'\Omega', 'ℂ': r'\mathbb{C}', 'ℝ': r'\mathbb{R}',
    'ℕ': r'\mathbb{N}', 'ℤ': r'\mathbb{Z}', '∇': r'\nabla',
    '∧': r'\wedge', '∨': r'\vee', '∀': r'\forall', '∃': r'\exists',
    '⇒': r'\Rightarrow', '⇔': r'\Leftrightarrow', '↦': r'\mapsto',
    '\x00': '\x12', '\x01': '\x13', '\x02': r'[', '\x03': r']', '\x08': r'\{', '\x09': r'\}', '\x11': ')', '\x13': ')', '\x14': r'[', '\x15': r']', '\x1a': r'\{', '\x1b': r'\}', '\x1e': r'/',
})


def page_symbols(page, version=2):
    """Извлекает символы и горизонтальные линии из PDF."""
    chars = []
    for block in page.get_text('rawdict')['blocks']:
        for line in block.get('lines', []):
            for span in line['spans']:
                for c in span['chars']:
                    if c['c'] not in (' ', '\n', '\t'):
                        token = SYMBOLS.get(c['c'], c['c'])
                        if version >= 3:
                            token = {'µ': r'\mu', 'ζ': r'\zeta'}.get(c['c'], token)
                        if span['font'].startswith('CMEX'):
                            token = {'P': r'\sum', 'Q': r'\prod', 'R': r'\int', 'n': r'\{', 'o': r'\}', '\x12': r'(', '\x13': r')'}.get(c['c'], token)
                        if version >= 5:
                            if span['font'].startswith('MSBM') and token.isascii() and token.isalpha():
                                token = r'\mathbb{' + token + '}'
                            elif span['font'].startswith('CMSY'):
                                token = {'\\': r'\setminus', '{': r'\{', '}': r'\}'}.get(c['c'], token)
                        chars.append({'text': token, 'x0': c['bbox'][0], 'x1': c['bbox'][2], 'y': c['origin'][1], 'size': span['size'], 'font': span['font']})
    bars = []
    for drawing in page.get_drawings():
        for item in drawing['items']:
            if item[0] == 'l' and abs(item[1].y-item[2].y) < 0.5 and abs(item[1].x-item[2].x) > 2:
                bars.append((min(item[1].x,item[2].x), max(item[1].x,item[2].x), item[1].y))
    return chars, bars


def fraction_region(chars, bars, parent):
    """Expand a fraction through connected, narrower child rules.

    Baselines of a nested denominator/root may lie beyond the outer rule's
    initial band. Child-rule membership grows the region without claiming a
    disconnected equation merely because it shares the same x coordinates.
    """
    x0, x1, y = parent
    parent_chars=[c for c in chars if x0-.5 <= (c['x0']+c['x1'])/2 <= x1+.5]
    parent_size=max((c['size'] for c in parent_chars),default=0)
    has_child=any(b!=parent and b[0]>=x0-.5 and b[1]<=x1+.5 and b[1]-b[0]<x1-x0-.25 for b in bars)
    band=1.8 if has_child else 1.2
    inside = [c for c in chars if x0-.5 <= (c['x0']+c['x1'])/2 <= x1+.5
              and abs(c['y']-y) < max(18 if has_child else 12, parent_size*band)]
    children = []
    candidates = [b for b in bars if b != parent and b[0] >= x0-.5
                  and b[1] <= x1+.5 and b[1]-b[0] < x1-x0-.25]
    changed = True
    while changed:
        changed = False
        for bar in candidates:
            if bar in children:
                continue
            if not any(bar[0]-.5 <= (c['x0']+c['x1'])/2 <= bar[1]+.5
                       and abs(c['y']-bar[2]) < max(12,c['size']*1.3) for c in inside):
                continue
            children.append(bar)
            for c in chars:
                radical = (c['text'] == '√' or c.get('font','').startswith('CMEX') and c['text'] in ('p','q','r','s'))
                linked_root = radical and abs(c['x1']-bar[0]) < 2 and abs(c['y']-bar[2]) < 2
                covered = bar[0]-.5 <= (c['x0']+c['x1'])/2 <= bar[1]+.5 and abs(c['y']-bar[2]) < max(12,c['size']*1.3)
                if c not in inside and (covered or linked_root):
                    inside.append(c)
                    changed = True
    return inside, children


def parse_formula(chars, bars, version=1, _complex=None):
    """Восстанавливает дроби и индексы по расположению символов."""
    complex_mode=len(bars)>=3 if _complex is None else _complex
    chars = [dict(c) for c in chars]
    if not chars:
        return ''
    if version >= 6:
        for factorial in list(chars):
            if factorial['text']!='!':
                continue
            previous=[c for c in chars if c is not factorial and c['x1']<=factorial['x0']+1
                      and 0<=factorial['x0']-c['x1']<max(8,factorial['size']*1.2)]
            if previous:
                base=max(previous,key=lambda c:c['x1'])
                factorial['y']=base['y']
    if version >= 6 and complex_mode:
        ordinary=[c for c in chars if not c.get('font','').startswith('CMEX')]
        if ordinary:
            largest=max(c['size'] for c in ordinary)
            bases=[c for c in ordinary if c['size']>=largest*.9 and c['text'] not in ('=','<','>')]
            if bases:
                baseline=Counter(round(c['y'],1) for c in bases).most_common(1)[0][0]
                relations={'=','<','>','\\leq','\\geq','\\neq','\\equiv','.',',',';',':','!'}
                for c in chars:
                    if c['text'] in relations and c['size']>=largest*.9 and 2<abs(c['y']-baseline)<c['size']*.9:
                        c['y']=baseline
    if version >= 6:
        for mark in list(chars):
            if mark['text'] in (r'\prime', '′'):
                candidates = [c for c in chars if c is not mark and c['text'] not in (r'\prime', '′') and c['x1'] <= mark['x0']+1 and 0 <= mark['x0']-c['x1'] < max(7, mark['size']*0.8) and abs(c['y']-mark['y']) < mark['size']*0.8]
                if candidates:
                    target = max(candidates, key=lambda c: c['x1'])
                    if target['text'].endswith(r'^{\prime}'):
                        target['text'] = target['text'][:-1] + r'\prime}'
                    else:
                        target['text'] += r'^{\prime}'
                    chars.remove(mark)
        for mark in list(chars):
            if mark['text'] in ('̸', '˙'):
                choices = [c for c in chars if c is not mark and abs(c['y']-mark['y']) < 3 and abs(c['x0']-mark['x0']) < mark['size']*0.7]
                if mark['text'] == '̸':
                    choices = [c for c in choices if c['text'] in ('=', r'\in', r'\subset', r'\supset', r'\equiv', '<', '>')]
                if choices:
                    target = min(choices, key=lambda c: abs(c['x0']-mark['x0']))
                    target['text'] = (r'\not ' + target['text']+' ') if mark['text'] == '̸' else r'\dot{'+target['text']+'}'
                    chars.remove(mark)
        pieces = [c for c in chars if c['text'] in ('\uf8eb', '\uf8ed', '\uf8f6', '\uf8f8')]
        chars = [c for c in chars if c not in pieces]
        for x in sorted({round(c['x0'], 1) for c in pieces}):
            group = [c for c in pieces if abs(c['x0']-x) < 0.1]
            anchor = dict(group[0])
            anchor['text'] = '\x12' if group[0]['text'] in ('\uf8eb', '\uf8ed') else '\x13'
            anchor['y'] = sum(c['y'] for c in group)/len(group)
            chars.append(anchor)
    if version >= 6:
        cases = extract_cases(chars, bars, lambda cs, bs: parse_formula(cs, bs, version))
        if cases is not None:
            return cases
        array = extract_array(chars, bars, lambda cs, bs: parse_formula(cs, bs, version))
        if array is not None:
            return array
    if version >= 6:
        for bar in sorted(bars, key=lambda b: b[1]-b[0], reverse=True):
            x0, x1, y = bar
            root_bar = any((c['text'] == '√' or c.get('font', '').startswith('CMEX') and c['text'] in ('p', 'q', 'r', 's')) and abs(c['x1']-x0) < 2 and abs(c['y']-y) < 2 for c in chars)
            if root_bar:
                continue
            inside, other = fraction_region(chars, bars, bar)
            above = [c for c in inside if c['y'] < y]
            below = [c for c in inside if c['y'] > y+1]
            if not above or not below:
                continue
            above_text = ''.join(c['text'] for c in sorted(above, key=lambda c: c['x0'])).replace(' ', '')
            if above_text == 'lim' and any(c['text'] == r'\to' for c in below) and any(c['text'] == r'\infty' for c in below):
                # TeX draws a short rule beneath a limits operator. Its lower
                # bound is an operator limit, not a fraction denominator.
                continue
            top_bars = [b for b in other if b[2] < y]
            bottom_bars = [b for b in other if b[2] > y]
            tex = r'\frac{'+parse_formula(above, top_bars, version)+'}{'+parse_formula(below, bottom_bars, version)+'}'
            local_size = max(c['size'] for c in inside)
            # TeX's fraction axis belongs to the surrounding math style, not
            # the smaller numerator glyphs. A sibling on that axis supplies
            # the real baseline (also when the PDF coordinates are scaled).
            anchor = None
            if complex_mode:
                siblings = [c for c in chars if c not in inside and not c.get('font','').startswith('CMEX')
                            and c['text'] not in ('=', '<', '>', '.', ',', ';', ':', '!')
                            and c['size'] >= local_size*.9]
                local = [c for c in siblings if max(x0-c['x1'],c['x0']-x1,0) <= 25]
                candidates = local or siblings
                axis_candidates=[c for c in siblings if c['y']>y
                                 and abs(c['y']-y-c['size']*.25)<max(.5,c['size']*.15)]
                if axis_candidates:
                    anchor=min(axis_candidates,key=lambda c:abs(c['y']-y-c['size']*.25))
                if candidates and anchor is None:
                    baselines = Counter(round(c['y'],1) for c in candidates)
                    baseline = max(baselines, key=lambda value:(baselines[value],-abs(value-(y+local_size*.9))))
                    anchor = min((c for c in candidates if round(c['y'],1)==baseline),
                                 key=lambda c:abs(c['x0']-(x1 if c['x0']>=x1 else x0)))
            else:
                siblings = [c for c in chars if c not in inside and c['y'] > y
                            and abs(c['y']-y-c['size']*.25) < max(.5,c['size']*.15)]
                anchor = min(siblings,key=lambda c:abs(c['y']-y-c['size']*.25)) if siblings else None
            replacement = {'text': tex, 'x0': x0, 'x1': x1,
                           'y': anchor['y'] if anchor else y+local_size*.25,
                           'size': anchor['size'] if anchor else local_size}
            return parse_formula([c for c in chars if c not in inside]+[replacement], [b for b in bars if b != bar and b not in other], version, complex_mode)
    if version >= 4:
        for root in list(chars):
            if (root.get('font', '').startswith('CMEX') and root['text'] in ('p', 'q', 'r', 's')) or root['text'] == '√':
                candidates = [b for b in bars if abs(b[0]-root['x1']) < 2 and abs(b[2]-root['y']) < 2]
                if not candidates:
                    continue
                bar = candidates[0]
                boundaries = [b[2] for b in bars if b != bar and b[2] > bar[2]+3 and b[0] <= bar[0]+1 and b[1] >= bar[1]-1]
                bottom = min(boundaries) if boundaries else bar[2]+45
                content = [c for c in chars if c is not root and bar[0]-0.5 <= (c['x0']+c['x1'])/2 <= bar[1]+0.5 and bar[2] < c['y'] < bottom]
                if not content:
                    continue
                degree = [c for c in chars if c is not root and c not in content and root['x0'] <= c['x0'] < root['x1'] and c['size'] < root['size']*0.7]
                inner_bars = [b for b in bars if b != bar and b[0] >= bar[0]-1 and b[1] <= bar[1]+1 and bar[2] < b[2] < bottom]
                tex = r'\sqrt' + ('['+parse_formula(degree, [], version)+']' if degree else '')
                tex += '{'+parse_formula(content, inner_bars, version)+'}'
                chars = [c for c in chars if c is not root and c not in content and c not in degree]
                chars.append({**root, 'text': tex, 'x1': bar[1], 'font': '', 'y': Counter(round(c['y'], 1) for c in content if c['size'] >= root['size']*0.9).most_common(1)[0][0]})
                bars = [b for b in bars if b != bar and b not in inner_bars]
        for accent in list(chars):
            if accent['text'] == '¯':
                candidates = [c for c in chars if c is not accent and abs(c['y']-accent['y']) < 3 and abs(c['x0']-accent['x0']) < 5]
                if candidates:
                    base = min(candidates, key=lambda c: abs(c['x0']-accent['x0']))
                    base['text'] = r'\bar{' + base['text'] + '}'
                    chars.remove(accent)
    if version >= 2:
        for x0, x1, y in sorted(bars, key=lambda b: b[1]-b[0]):
            inside = [c for c in chars if x0-0.5 <= (c['x0']+c['x1'])/2 <= x1+0.5 and abs(c['y']-y) < 35]
            above = [c for c in inside if c['y'] < y+1]
            below = [c for c in inside if c['y'] > y+1]
            above_text = ''.join(c['text'] for c in sorted(above, key=lambda c: c['x0'])).replace(' ', '')
            if above_text == 'lim' and any(c['text'] == r'\to' for c in below) and any(c['text'] == r'\infty' for c in below):
                continue
            if version >= 4 and below and not above:
                tex = r'\overline{' + parse_formula(below, [], version) + '}'
                chars = [c for c in chars if c not in below]
                chars.append({'text': tex, 'x0': x0, 'x1': x1, 'y': max(c['y'] for c in below), 'size': max(c['size'] for c in below)})
            if above and below:
                tex = r'\frac{' + parse_formula(above, [], version) + '}{' + parse_formula(below, [], version) + '}'
                chars = [c for c in chars if c not in inside]
                chars.append({'text': tex, 'x0': x0, 'x1': x1, 'y': y+3, 'size': max(c['size'] for c in inside)})
    if version >= 5:
        if version >= 6:
            verticals = [c for c in chars if c.get('font', '').startswith('CMEX') and c['text'] == '\x0c']
            chars = [c for c in chars if c not in verticals]
            for x in sorted({round(c['x0'], 1) for c in verticals}):
                near = [c for c in verticals if abs(c['x0']-x) < 0.1]
                chars.append({**near[0], 'text': '|', 'font': '', 'y': 0})
        ordinary = [c for c in chars if not c.get('font', '').startswith('CMEX')] or chars
        size = max(c['size'] for c in ordinary)
        main_y = Counter(round(c['y'], 1) for c in ordinary if c['size'] >= size*0.9).most_common(1)[0][0]
        for c in list(chars):
            if c['text'] == '|' and c['y'] == 0:
                c['y'] = main_y
            if c.get('font', '').startswith('CMEX'):
                if c['text'] in ('\x12', '\x13', '\x10', '\x11'):
                    c['text'] = '(' if c['text'] in ('\x12', '\x10') else ')'
                    c['y'] = main_y
                elif c['text'] in (r'[', r']', r'\{', r'\}'):
                    c['y'] = main_y
                elif c['text'] in ('Z', 'I', 'X', 'Y', r'\int', r'\oint', r'\sum', r'\prod'):
                    operator = {'Z': r'\int', 'I': r'\oint', 'X': r'\sum', 'Y': r'\prod'}.get(c['text'], c['text'])
                    left_limit_extent = max(12, c['size']*1.8) if version >= 6 else 2
                    limit_offset=max(2.5,size*.4) if version >= 6 and complex_mode else 7
                    limits = [a for a in chars if a is not c and a['size'] < size*0.9 and c['x0']-left_limit_extent <= a['x0'] <= c['x1']+(12 if version >= 6 else 5) and abs(a['y']-main_y) > limit_offset]
                    lower = [a for a in limits if a['y'] > main_y]
                    upper = [a for a in limits if a['y'] < main_y]
                    if version >= 6:
                        c['text'] = operator
                        c['subscript'] = parse_formula(lower, [], version)
                        c['superscript'] = parse_formula(upper, [], version)
                    else:
                        c['text'] = operator + ('_{'+parse_formula(lower, [], version)+'}' if lower else '') + ('^{'+parse_formula(upper, [], version)+'}' if upper else '')
                    c['y'] = main_y
                    chars = [a for a in chars if a not in limits]
        if version >= 6:
            for arrow in list(chars):
                if arrow['text'] == r'\to':
                    stems = [c for c in chars if c['text'] == '-' and abs(c['y']-arrow['y']) < 1 and 0 < arrow['x0']-c['x0'] < c['size']]
                    if stems:
                        arrow['text'] = r'\longrightarrow'
                        chars = [c for c in chars if c not in stems]
        ordinary_chars = [c for c in chars if c['size'] >= size*0.9]
        ordered = sorted((c for c in ordinary_chars if abs(c['y']-main_y) < 1), key=lambda c: c['x0'])
        for word in ('sin', 'cos', 'lim', 'inf', 'exp', 'log', 'res', 'max', 'min'):
            for i in range(len(ordered)-len(word)+1):
                group = ordered[i:i+len(word)]
                if ''.join(c['text'] for c in group) == word and max(c['y'] for c in group)-min(c['y'] for c in group) < 1:
                    group[0]['text'] = r'\operatorname{res}' if word == 'res' else '\\'+word
                    if word in ('lim', 'inf', 'res', 'max', 'min'):
                        left_extent = max(2, size*1.8) if word in ('res', 'max', 'min', 'inf') else max(2, size*0.5)
                        full_size_limit = word in ('max', 'min')
                        lower = [c for c in chars if c not in group
                                 and (full_size_limit or c['size'] < size*0.9)
                                 and group[0]['x0']-left_extent <= c['x0'] <= group[-1]['x1']+(left_extent if full_size_limit else 2)
                                 and c['y'] > group[0]['y']+3]
                        if lower:
                            if version >= 6:
                                lower_bars = [bar for bar in bars if min(c['x0'] for c in lower)-1 <= bar[0]
                                              and bar[1] <= max(c['x1'] for c in lower)+1
                                              and bar[2] > group[0]['y']+2]
                                group[0]['subscript'] = parse_formula(lower, lower_bars, version)
                            else:
                                group[0]['text'] += '_{' + parse_formula(lower, [], version) + '}'
                            chars = [c for c in chars if c not in lower]
                    group[0]['x1'] = group[-1]['x1']
                    chars = [c for c in chars if c not in group[1:]]
        if version >= 6:
            for overline in list(chars):
                if not overline['text'].startswith(r'\overline{') or r'\lim' not in overline['text']:
                    continue
                leading_n = [c for c in chars if c is not overline and c['text'] == 'n'
                             and c['size'] < size*0.9 and c['x0'] < overline['x0']
                             and overline['x0']-c['x0'] <= max(12, size*1.8)
                             and c['y'] > main_y+3]
                if not leading_n:
                    continue
                match = re.search(r'\\lim\s*_\s*\{', overline['text'])
                if match:
                    offset = match.end()
                    overline['text'] = overline['text'][:offset]+'n'+overline['text'][offset:]
                else:
                    overline['text'] = overline['text'].replace(r'\lim', r'\lim_{n}', 1)
                overline['y'] = main_y
                chars = [c for c in chars if c not in leading_n]
    if version >= 6:
        return serialize(chars, lambda cs, bs: parse_formula(cs, bs, version))
    largest = max(c['size'] for c in chars)
    base_chars = [c for c in chars if c['size'] >= largest * 0.9] if version >= 3 else chars
    baseline = Counter(round(c['y'], 1) for c in base_chars).most_common(1)[0][0]
    result = ''
    group = ''
    mode = None
    for c in sorted(chars, key=lambda c: (c['x0'], c['y'])):
        threshold = 1.2 if version >= 3 and c['size'] < largest * 0.9 else 2
        current = '^' if c['y'] < baseline-threshold else '_' if c['y'] > baseline+threshold else None
        if current != mode:
            if mode:
                result += mode + '{' + group + '}'
                group = ''
            mode = current
        token = c['text'] + (' ' if c['text'].startswith('\\') and c['text'][1:].isalpha() else '')
        if mode:
            group += token
        else:
            result += token
    if mode:
        result += mode + '{' + group + '}'
    return result
