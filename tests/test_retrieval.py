import unittest

from enterprise_retrieval.embedding import source_windows
from enterprise_retrieval.index import PAYLOAD_TOKENS, prefix_coverage, token_windows
from enterprise_retrieval.rerank import order_candidates
from enterprise_retrieval.search import fuse


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
    def test_maximum_uses_all_chunks_and_preserves_document_order(self):
        import torch

        from enterprise_retrieval.search import document_maxima

        scores = torch.tensor([[0.1, 0.9, 0.7, 0.6, 0.6], [1.0, 0.2, 0.3, 0.4, 0.8]])
        parents = torch.tensor([0, 0, 1, 2, 2])
        expected = torch.tensor([[0.9, 0.7, 0.6], [1.0, 0.3, 0.8]])
        self.assertTrue(torch.equal(document_maxima(scores, parents, 3), expected))

    def test_best_documents_skips_repeated_chunks_without_merging_ids(self):
        from enterprise_retrieval.search import best_documents

        hits = [{"corpus_row": 1, "chunk_row": i} for i in range(300)]
        hits += [{"corpus_row": i, "chunk_row": 300 + i} for i in range(2, 60)]
        selected = best_documents(hits)
        self.assertEqual([h["corpus_row"] for h in selected], list(range(1, 51)))
        self.assertEqual(selected[0]["chunk_row"], 0)


class QwenEmbeddingTests(unittest.TestCase):
    def test_last_real_token_with_either_padding_direction(self):
        import torch

        from enterprise_retrieval.embedding import last_token_pool

        hidden = torch.arange(24).reshape(2, 4, 3)
        masks = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
        self.assertTrue(torch.equal(last_token_pool(hidden, masks), hidden[[0, 1], [3, 1]]))
        with self.assertRaises(ValueError):
            last_token_pool(hidden, torch.zeros_like(masks))

    def test_original_text_is_partitioned_without_decoding_bge_tokens(self):
        documents = [{"title": "ABC", "content": "déf!"}, {"title": "", "content": ""}]
        chunks = [
            {"chunk_row": 1, "corpus_row": 1, "char_start": 0, "char_end": 5},
            {"chunk_row": 2, "corpus_row": 1, "char_start": 5, "char_end": 8},
            {"chunk_row": 3, "corpus_row": 2, "char_start": 0, "char_end": 1},
        ]
        self.assertEqual(
            list(source_windows(documents, chunks)), [(1, 1, "ABC\nd"), (2, 1, "éf!"), (3, 2, "\n")]
        )
        with self.assertRaises(ValueError):
            list(source_windows(documents, [{**chunks[0], "char_start": 1}]))
        with self.assertRaises(ValueError):
            list(source_windows(documents[:1], chunks[:1]))


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


def document(row, score=0):
    return dict(
        corpus_row=row,
        doc_id="colliding-id",
        source_type="slack",
        title="title",
        content=str(row),
        score=score,
    )


class EmbeddingTests(unittest.TestCase):
    def test_overlap_fuses_rows_but_preserves_colliding_ids(self):
        result = fuse([document(1), document(2)], [document(3), document(2)])
        self.assertEqual([r["corpus_row"] for r in result], [2, 1, 3])
        self.assertAlmostEqual(result[0]["score"], 2 / 62)
        self.assertEqual(result[0]["lexical_rank"], 2)
        self.assertEqual(result[0]["dense_rank"], 2)
        self.assertEqual(len(fuse([document(1)], [document(2)], limit=1)), 1)

    def test_fusion_rejects_repeated_or_changed_rows(self):
        with self.assertRaises(ValueError):
            fuse([document(1), document(1)], [])
        with self.assertRaises(ValueError):
            fuse([document(1)], [{**document(1), "content": "changed"}])
