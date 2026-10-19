import unittest

from enterprise_retrieval.index import PAYLOAD_TOKENS, prefix_coverage, token_windows
from enterprise_retrieval.rerank import order_candidates


class ChunkTests(unittest.TestCase):
    def test_every_token_and_character_has_exactly_one_window(self):
        ids = list(range(1023))
        offsets = [(2 * i, 2 * i + 1) for i in ids]
        windows = list(token_windows(ids, offsets, 2046))
        self.assertEqual([token for (*_, group) in windows for token in group], ids)
        self.assertTrue(all((len(w[-1]) + 2 <= 512 for w in windows)))
        self.assertEqual([len(w[-1]) for w in windows], [PAYLOAD_TOKENS, PAYLOAD_TOKENS, 3])
        self.assertEqual(windows[0][2], 0)
        self.assertEqual(windows[-1][3], 2046)
        self.assertTrue(all((a[3] == b[2] for (a, b) in zip(windows, windows[1:]))))

    def test_empty_document_and_exact_boundary(self):
        self.assertEqual(list(token_windows([], [], 3)), [(0, 0, 0, 3, [])])
        self.assertEqual(len(list(token_windows([1] * 510, [(0, 1)] * 510, 1))), 1)
        with self.assertRaises(ValueError):
            list(token_windows([1], [], 1))

    def test_loss_uses_full_corpus_tokens_and_both_old_cutoffs(self):
        offsets = [(i, i + 1) for i in range(1000)]
        result = prefix_coverage(offsets, offsets[:800], title_length=9, content_length=990)
        self.assertEqual(result["full_payload_tokens"], 1000)
        self.assertEqual(result["old_visible_full_tokens"], 510)
        self.assertEqual(result["old_visible_content_characters"], 500)
        shorter = prefix_coverage(offsets, offsets[:200], title_length=9, content_length=990)
        self.assertEqual(shorter["old_visible_full_tokens"], 200)


class ChunkRetrievalTests(unittest.TestCase):
    def test_best_documents_skips_repeated_chunks_without_merging_ids(self):
        from enterprise_retrieval.search import best_documents

        hits = [{"corpus_row": 1, "chunk_row": i} for i in range(300)]
        hits += [{"corpus_row": i, "chunk_row": 300 + i} for i in range(2, 60)]
        selected = best_documents(hits)
        self.assertEqual([h["corpus_row"] for h in selected], list(range(1, 51)))
        self.assertEqual(selected[0]["chunk_row"], 0)


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
