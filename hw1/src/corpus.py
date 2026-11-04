"""Deterministic corpus I/O and splitting outside LaTeX math regions."""
import hashlib
import json
import re
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def split_text(text, target_chars):
    """Pack paragraphs; a math expression is indivisible, even above target."""
    if target_chars < 1:
        raise ValueError('target_chars must be positive')
    # Preserve every input character. Split only at paragraph boundaries outside
    # $...$/$$...$$; escaped dollar signs do not change math state.
    units, start, math = [], 0, None
    for match in re.finditer(r'(?<!\\)\$\$?|\n\s*\n', text):
        token = match.group()
        if token.startswith('$'):
            if math is None:
                math = token
            elif math == token:
                math = None
        elif math is None:
            units.append(text[start:match.end()])
            start = match.end()
    units.append(text[start:])
    pieces, pending = [], ''
    for unit in units:
        if pending and len(pending) + len(unit) > target_chars:
            pieces.append(pending)
            pending = ''
        pending += unit
    if pending:
        pieces.append(pending)
    return pieces


def validate_corpus(rows):
    seen = set()
    for row in rows:
        if not isinstance(row.get('block_id'), str) or not row['block_id']:
            raise ValueError('Missing/non-string block_id')
        if row['block_id'] in seen:
            raise ValueError('Duplicate block_id: ' + row['block_id'])
        seen.add(row['block_id'])
        if type(row.get('page')) is not int or row['page'] < 1:
            raise ValueError('Invalid PDF page')
        if not isinstance(row.get('section'), str):
            raise ValueError('section must be a string')
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Empty/non-string block text')
    if not rows:
        raise ValueError('Empty corpus')
