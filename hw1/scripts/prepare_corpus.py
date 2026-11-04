"""Build the local corpus from the source PDF without reading QA or gold."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pymupdf
from src.corpus import sha256, split_text, validate_corpus, write_jsonl
from src.extraction import extract_page


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows, section, diagnostics = [], '', []
    with pymupdf.open(args.pdf) as document:
        for page in document:
            extracted = extract_page(page)
            pieces = split_text(extracted['text'], config['corpus']['target_chars'])
            for index, text in enumerate(pieces, 1):
                heading = re.search(r'(?:^|\n)\s*(§\s*\d+\.[^\n]{0,180})', text)
                if heading:
                    section = heading[1].strip()
                if text.strip():
                    rows.append({'block_id': f'p{page.number+1:03d}_b{index:03d}',
                                 'page': page.number+1, 'section': section, 'text': text})
            diagnostics.append({'page': page.number+1,
                                'formula_count': len(extracted['formulas']),
                                'flagged_formulas': sum(bool(f['warnings']) for f in extracted['formulas'])})
    validate_corpus(rows)
    write_jsonl(args.output, rows)
    metadata = {'origin': 'local_pdf_extraction', 'format_version': 1,
                'pdf_sha256': sha256(args.pdf), 'corpus_sha256': sha256(args.output),
                'extractor_sha256': {name: sha256(ROOT / name) for name in
                    ['src/extraction.py', 'src/formulas.py', 'src/formula_layout.py', 'src/corpus.py', 'scripts/prepare_corpus.py']},
                'target_chars': config['corpus']['target_chars'],
                'pages': len(diagnostics), 'blocks': len(rows), 'diagnostics': diagnostics}
    args.output.with_suffix('.meta.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: metadata[k] for k in ['pages', 'blocks', 'corpus_sha256']}))


if __name__ == '__main__':
    main()
