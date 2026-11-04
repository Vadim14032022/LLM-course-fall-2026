import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval import Retriever, reciprocal_rank_fusion


class RetrievalTests(unittest.TestCase):
    def test_seed_results_precede_neighbors_before_budget_trimming(self):
        retriever = object.__new__(Retriever)
        retriever.config = {'retrieval': {'neighbors': 1, 'neighbor_policy':'seeds_first'}}
        retriever.method = 'bm25'
        retriever.chunks = [{'chunk_id':str(i), 'block_id':'s'+str(i)} for i in range(5)]
        retriever.bm25 = SimpleNamespace(get_scores=lambda q: [0.,5.,1.,4.,0.])
        hits = retriever.search({'question':'test','options':[]}, 2)
        self.assertEqual([h['chunk_id'] for h in hits[:2]], ['1','3'])
        self.assertTrue(all(h['neighbor_of'] is None for h in hits[:2]))
        self.assertEqual(len({h['chunk_id'] for h in hits}), len(hits))

    def test_rrf_consensus(self):
        scores = reciprocal_rank_fusion([[0, 1, 2], [2, 1, 0]], k=1)
        self.assertAlmostEqual(scores[0], scores[2])
        self.assertGreater(scores[0], scores[1])

    def test_neighbors_are_unique_and_keep_original_ids(self):
        retriever = object.__new__(Retriever)
        retriever.config = {'retrieval': {'neighbors': 1}}
        retriever.method = 'bm25'
        retriever.chunks = [{'chunk_id': str(i), 'block_id': 'external-'+str(i)} for i in range(3)]
        retriever.bm25 = SimpleNamespace(get_scores=lambda q: [1., 3., 2.])
        hits = retriever.search({'question': 'test', 'options': []}, 2)
        self.assertEqual([h['block_id'] for h in hits], ['external-1', 'external-2', 'external-0'])
        self.assertEqual(len({h['chunk_id'] for h in hits}), 3)


if __name__ == '__main__':
    unittest.main()
