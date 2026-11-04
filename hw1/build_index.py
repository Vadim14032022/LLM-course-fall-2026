"""Build a portable lexical index without generation or question access."""
import argparse
import json
import resource
import time
from pathlib import Path
from src.corpus import read_jsonl, write_jsonl, validate_corpus, split_text, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    rows = read_jsonl(args.corpus)
    validate_corpus(rows)
    chunks = []
    for order, row in enumerate(rows):
        for number, text in enumerate(split_text(row['text'], config['chunking']['target_chars'])):
            chunks.append({**row, 'chunk_id': f'{order}:{number}', 'source_order': order, 'text': text})
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / 'blocks.jsonl', rows)
    write_jsonl(args.output / 'chunks.jsonl', chunks)
    embedding = None
    if config['retrieval']['method'] in ('dense', 'hybrid'):
        import numpy as np
        from src.dense import DenseEncoder
        embedding = config['embedding']
        encoder = DenseEncoder(embedding)
        vectors = encoder.encode([c['text'] for c in chunks], 'passage')
        np.save(args.output / 'embeddings.npy', vectors, allow_pickle=False)
    manifest = {'format_version': 1, 'corpus_sha256': sha256(args.corpus),
                'lexical_tokenizer_sha256_at_build': sha256(Path(__file__).resolve().parent/'src/math_search.py'),
                'chunking': config['chunking'], 'embedding': embedding, 'blocks': len(rows), 'chunks': len(chunks),
                'build_seconds': time.perf_counter()-started,
                'peak_ram_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
