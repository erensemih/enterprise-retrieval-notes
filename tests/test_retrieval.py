import unittest

from enterprise_retrieval.rerank import order_candidates


class RerankingTests(unittest.TestCase):
    def test_scores_reorder_records_without_merging_document_ids(self):
        candidates = [
            {"doc_id": "same", "corpus_row": 1, "content": "first", "score": -5},
            {"doc_id": "same", "corpus_row": 2, "content": "second", "score": -4},
            {"doc_id": "other", "corpus_row": 3, "content": "third", "score": -3},
        ]
        ranked = order_candidates(candidates, [1.0, 2.0, 2.0])
        self.assertEqual([r["corpus_row"] for r in ranked], [2, 3, 1])
        for record in ranked:
            original = candidates[record["bm25_rank"] - 1]
            self.assertEqual({k: record[k] for k in original}, original)
        self.assertFalse(any(("reranker_score" in record for record in candidates)))

    def test_incomplete_or_nonfinite_scores_are_rejected(self):
        for scores in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                order_candidates([{"corpus_row": 1}], scores)
        self.assertEqual(order_candidates([], []), [])
