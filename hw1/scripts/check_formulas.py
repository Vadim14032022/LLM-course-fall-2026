import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.formulas import page_symbols, parse_formula


def normalize(tex):
    """Убирает команды пробелов и необязательный размер скобок."""
    tex = re.sub(r'\\bar\s+([A-Za-z0-9])', r'\\bar{\1}', tex)
    tex = re.sub(r'\\(?:left|right|quad|qquad)\b|\\[,;! ]', '', tex)
    tex = re.sub(r'([_^])([A-Za-z0-9])', r'\1{\2}', tex)
    tex = re.sub(r'([_^])(\\[A-Za-z]+)', r'\1{\2}', tex)
    return re.sub(r'\s+', '', tex)


def evaluate(pdf, manifest, output, version):
    """Сравнивает извлечённый LaTeX с эталонами после независимого парсинга."""
    with pymupdf.open(pdf) as doc:
        predictions = [parse_formula(*page_symbols(page, version), version=version) for page in doc]
    gold = json.loads(manifest.read_text(encoding='utf-8'))
    if len(predictions) != len(gold):
        raise ValueError('Число страниц не совпало с числом формул')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item, prediction in zip(gold, predictions):
        expected, actual = normalize(item['tex']), normalize(prediction)
        rows.append({'id': item['id'], 'expected': item['tex'], 'actual': prediction,
                     'exact': expected == actual,
                     'similarity': difflib.SequenceMatcher(None, expected, actual).ratio()})
        (output / (item['id'] + '.tex')).write_text(prediction + '\n', encoding='utf-8')
    report = {'version': version, 'exact': sum(r['exact'] for r in rows), 'total': len(rows), 'rows': rows}
    (output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Точное совпадение: {report['exact']}/{report['total']}")
    for row in rows:
        if not row['exact']:
            print(row['id'], repr(row['actual']))


def main():
    """Запускает проверку выбранной версии парсера."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', type=int, choices=[1, 2, 3, 4, 5, 6], default=6)
    args = parser.parse_args()
    evaluate(args.pdf, args.manifest, args.output, args.version)


if __name__ == '__main__':
    main()
