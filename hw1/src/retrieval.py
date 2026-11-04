"""CPU BM25 search over immutable chunks with original source IDs."""
import json
from rank_bm25 import BM25Okapi
from .corpus import read_jsonl
from .math_search import search_tokens


def tokenize(text):
    return search_tokens(text)


class Retriever:
    def __init__(self, index, config=None):
        from pathlib import Path
        self.chunks = read_jsonl(Path(index) / 'chunks.jsonl')
        self.bm25 = BM25Okapi([tokenize(c['text']) or ['_'] for c in self.chunks])
        self.config = config or {'retrieval': {'method': 'bm25', 'neighbors': 0}}
        self.method = self.config['retrieval']['method']
        if self.method not in ('bm25', 'dense', 'hybrid'):
            raise ValueError('Unknown retrieval method: ' + self.method)
        if self.method in ('dense', 'hybrid'):
            import numpy as np
            from .dense import DenseEncoder
            manifest = json.loads((Path(index)/'manifest.json').read_text())
            if manifest.get('embedding') != self.config['embedding']:
                raise ValueError('Embedding configuration/index mismatch: rebuild index')
            self.encoder = DenseEncoder(self.config['embedding'])
            self.vectors = np.load(Path(index)/'embeddings.npy', allow_pickle=False)
            if len(self.vectors) != len(self.chunks):
                raise ValueError('Embedding/chunk count mismatch')

    def search(self, question, top_k=6):
        query = question['question'] + '\n' + '\n'.join(o['text'] for o in question['options'])
        scores = self.bm25.get_scores(tokenize(query))
        lexical = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        if self.method in ('dense', 'hybrid'):
            semantic = self.vectors @ self.encoder.encode([query], 'query')[0]
            dense_ranked = sorted(range(len(semantic)), key=lambda i: (-semantic[i], i))
            if self.method == 'dense':
                scores, lexical = semantic, dense_ranked
            else:
                scores = reciprocal_rank_fusion([lexical, dense_ranked], self.config['retrieval'].get('rrf_k', 60))
                lexical = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        ranked = lexical[:top_k]
        radius = self.config['retrieval'].get('neighbors', 0)
        expanded, seen = [], set()
        policy = self.config['retrieval'].get('neighbor_policy', 'interleaved')
        if policy not in ('interleaved', 'seeds_first'):
            raise ValueError('Unknown neighbor policy: ' + policy)
        if policy == 'seeds_first':
            for i in ranked:
                seen.add(i)
                expanded.append({**self.chunks[i], 'score': float(scores[i]), 'neighbor_of': None})
        for seed in ranked:
            for i in [seed] + [j for distance in range(1, radius+1) for j in (seed+distance, seed-distance)]:
                if 0 <= i < len(self.chunks) and i not in seen:
                    seen.add(i)
                    expanded.append({**self.chunks[i], 'score': float(scores[i]), 'neighbor_of': self.chunks[seed]['chunk_id'] if i != seed else None})
        return expanded


def reciprocal_rank_fusion(rankings, k=60):
    scores = [0.0] * len(rankings[0])
    for ranking in rankings:
        for rank, index in enumerate(ranking, 1):
            scores[index] += 1/(k+rank)
    return scores
