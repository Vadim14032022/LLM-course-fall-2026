from collections import Counter


def split_rows(chars, tolerance=2):
    """Группирует символы по базовым строкам."""
    rows = []
    for c in sorted(chars, key=lambda c: c['y']):
        if not rows or abs(c['y']-rows[-1][0]['y']) > tolerance:
            rows.append([c])
        else:
            rows[-1].append(c)
    return rows


def extract_array(chars, bars, parse):
    """Восстанавливает прямоугольную матрицу по строкам и столбцам."""
    special = [c for c in chars if c.get('font', '').startswith('CMEX')]
    if not special or bars:
        return None
    inner = [c for c in chars if c not in special]
    rows = split_rows(inner)
    if len(rows) < 2:
        return None
    if not all(abs(c['size']-inner[0]['size']) < 1 for c in inner):
        return None
    rows = [sorted(row, key=lambda c: c['x0']) for row in rows]
    if len({len(row) for row in rows}) != 1 or len(rows[0]) < 2:
        return None
    if any(abs(row[i]['x0']-rows[0][i]['x0']) > 4 for row in rows for i in range(len(row))):
        return None
    content = r'\\'.join('&'.join(parse([c], []) for c in row) for row in rows)
    return r'\begin{pmatrix}'+content+r'\end{pmatrix}'


def serialize(chars, parse):
    """Собирает базовые символы и привязанные к ним вложенные индексы."""
    if not chars:
        return ''
    for c in chars:
        for delimiter in (r')', r'\}'):
            if c['text'] == delimiter+r'^{\prime}':
                c['text'] = delimiter
                c['superscript'] = c.get('superscript', '')+r'\prime'
    largest = max(c['size'] for c in chars)
    base = [c for c in chars if c['size'] >= largest*0.9]
    baseline = Counter(round(c['y'], 1) for c in base).most_common(1)[0][0]
    atoms = []
    pending = []
    mode = None
    for c in sorted(chars, key=lambda c: (c['x0'], c['y'])):
        threshold = 1.2 if c['size'] < largest*0.9 else 2
        delimiter = c.get('font', '').startswith('CMEX') and c['text'] in (r'\{', r'\}', r'(', r')')
        current = None if delimiter else '^' if c['y'] < baseline-threshold else '_' if c['y'] > baseline+threshold else None
        if current != mode and pending:
            if not atoms:
                atoms.append({'base': '', '^': '', '_': ''})
            value = parse(pending, [])
            if mode == '^' and atoms[-1]['base'].endswith(r'^{\prime}'):
                atoms[-1]['base'] = atoms[-1]['base'][:-1] + r' ' + value + '}'
            else:
                atoms[-1][mode] += value
            pending = []
        mode = current
        if mode:
            pending.append(c)
        else:
            token = c['text']
            token += ' ' if token.startswith('\\') and token[1:].isalpha() and not c.get('subscript') and not c.get('superscript') else ''
            atoms.append({'base': token, '^': c.get('superscript', ''), '_': c.get('subscript', '')})
    if pending:
        if not atoms:
            atoms.append({'base': '', '^': '', '_': ''})
        atoms[-1][mode] += parse(pending, [])
    if (len(atoms) > 1 and not atoms[0]['base'] and atoms[0]['_']
            and atoms[1]['base'].startswith(r'\lim')):
        left_limit = atoms.pop(0)['_'].rstrip()
        operator = atoms[0]
        operator['base'] = operator['base'].rstrip()
        operator['_'] = left_limit + (r'\ ' + operator['_'].strip() if operator['_'] else '')
    return ''.join(a['base']+('_{'+a['_']+'}' if a['_'] else '')+('^{'+a['^']+'}' if a['^'] else '') for a in atoms)


def extract_cases(chars, bars, parse):
    """Восстанавливает две колонки кусочно заданной функции."""
    braces = [c for c in chars if c.get('font', '').startswith('CMEX')
              and (c['text'] == r'\{' or c['text'] == '(' and not c.get('extensible')
                   and not any(d['text'] == ')' and d['x0'] > c['x0'] for d in chars))]
    if not braces or bars:
        return None
    brace = braces[0]
    body = [c for c in chars if c['x0'] > brace['x1']-1 and c is not brace]
    if not body:
        return None
    size = max(c['size'] for c in body)
    rows = split_rows([c for c in body if c['size'] > size*0.9])
    if len(rows) < 2:
        return None
    for c in body:
        if c['size'] <= size*0.9:
            min(rows, key=lambda row: abs(row[0]['y']-c['y'])).append(c)
    cells = []
    for row in rows:
        row.sort(key=lambda c: c['x0'])
        gaps = [(row[i+1]['x0']-row[i]['x1'], i+1) for i in range(len(row)-1)]
        if not gaps:
            return None
        gap, cut = max(gaps)
        if gap < size*0.7:
            return None
        cells.append(parse(row[:cut], [])+'&'+parse(row[cut:], []))
    prefix = [c for c in chars if c['x1'] <= brace['x0']]
    return parse(prefix, [])+r'\begin{cases}'+r'\\'.join(cells)+r'\end{cases}'
