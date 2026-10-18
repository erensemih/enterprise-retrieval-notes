import unittest

from enterprise_retrieval.evaluation import retrieval_metrics

"Protect metric denominators, record identity, and ungraded semantic outcomes."


class EvaluationTests(unittest.TestCase):
    def test_duplicate_results_do_not_inflate_recall_or_compress_rank(self):
        result = retrieval_metrics(["a", "b"], ["a", "a", "x"], set(), "basic", 5)
        self.assertEqual(result["recall"], 0.5)
        ranked = retrieval_metrics(["b"], ["x", "x", "b"], set(), "basic", 5)
        self.assertAlmostEqual(ranked["reciprocal_rank"], 1 / 3)

    def test_unavailable_reference_is_not_success_or_failure(self):
        for expected, ambiguous, kind in [
            ([], set(), "basic"),
            ([], set(), "info_not_found"),
            ([], set(), "high_level"),
            (["a"], {"a"}, "basic"),
        ]:
            with self.subTest(kind=kind, expected=expected):
                result = retrieval_metrics(expected, ["a"], ambiguous, kind, 5)
                self.assertIsNone(result["recall"])
                self.assertIsNone(result["reciprocal_rank"])

    def test_no_retrieval_hit_is_zero_when_a_reference_exists(self):
        result = retrieval_metrics(["a"], [], set(), "basic", 5)
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["reciprocal_rank"], 0)


if __name__ == "__main__":
    unittest.main()


class EvaluationRunnerTests(unittest.TestCase):
    def test_only_question_reaches_retrieval_and_na_stays_outside_denominator(self):
        import json
        import sqlite3
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock

        from enterprise_retrieval.evaluation import evaluate

        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "documents.sqlite"
            with sqlite3.connect(index) as db:
                db.execute("CREATE TABLE metadata (value TEXT)")
                db.execute(
                    "INSERT INTO metadata VALUES (?)", (json.dumps({"duplicate_records": []}),)
                )
            retriever = Mock(corpus_index=index)
            retriever.search.return_value = {"documents": [{"doc_id": "a"}]}
            questions = [
                {
                    "question_id": "one",
                    "question": "First?",
                    "question_type": "basic",
                    "expected_doc_ids": ["a"],
                    "gold_answer": "SECRET",
                },
                {
                    "question_id": "two",
                    "question": "Second?",
                    "question_type": "info_not_found",
                    "expected_doc_ids": [],
                },
            ]
            result = evaluate(retriever, questions)
            self.assertEqual(result["scored"], 1)
            self.assertEqual(result["unscored"], 1)
            self.assertEqual(result["reference_recall_at_5"], 1)
            self.assertEqual(
                [call.args[0] for call in retriever.search.call_args_list], ["First?", "Second?"]
            )
            self.assertNotIn("SECRET", str(retriever.search.call_args_list))
            with self.assertRaises(ValueError):
                evaluate(retriever, [questions[0], questions[0]])
