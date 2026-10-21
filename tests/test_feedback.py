import json
import unittest

from enterprise_retrieval.feedback import expansion_messages as messages
from enterprise_retrieval.feedback import followup_messages as build_messages
from enterprise_retrieval.feedback import (
    keywords,
    match_counts,
    parse_decision,
    parse_keywords,
    second_round,
    terms,
    weighted_fuse,
)


def document(row, content="Evidence"):
    return {
        "corpus_row": row,
        "doc_id": f"doc-{row}",
        "source_type": "test",
        "title": f"Document {row}",
        "content": content,
        "score": -1.0,
    }


class FollowupTests(unittest.TestCase):
    def test_only_five_clipped_sources_and_question_reach_model(self):
        record = {
            "question": "Who was affected?",
            "candidates": [document(i, "a" * 5000) for i in range(1, 51)],
            "gold_answer": "SECRET LABEL",
            "answer_facts": ["SECRET FACT"],
            "expected_doc_ids": ["SECRET ID"],
        }
        messages = build_messages(record)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(set(payload), {"original_question", "provided_sources"})
        self.assertEqual(len(payload["provided_sources"]), 5)
        self.assertEqual(len(payload["provided_sources"][0]["excerpt"]), 4500)
        self.assertTrue(payload["provided_sources"][0]["excerpt_clipped"])
        self.assertNotIn("SECRET", json.dumps(messages))
        self.assertNotIn("doc_id", messages[1]["content"])

    def test_stop_does_not_search_and_preserves_original_ranking(self):
        decision = parse_decision(
            json.dumps(
                {
                    "needs_more_evidence": False,
                    "assessment": "All requested information is present.",
                    "missing_information": [],
                    "follow_up_question": None,
                }
            )
        )
        original = [document(i) for i in range(50)]

        def forbidden_search(question):
            self.fail("The stop branch must never perform a second search")

        result = second_round({"candidates": original}, decision, forbidden_search)
        self.assertIs(result["candidates"], original)
        self.assertEqual(result["second_candidates"], [])

    def test_second_search_uses_followup_alone_and_merges_unique_rows(self):
        decision = parse_decision(
            json.dumps(
                {
                    "needs_more_evidence": True,
                    "assessment": "Cause is covered; customer impact is not.",
                    "missing_information": ["Affected customers"],
                    "follow_up_question": "Which customers were affected by the outage?",
                }
            )
        )
        observed = []

        def retrieve(question):
            observed.append(question)
            return ([document(i) for i in (*range(50, 75), *range(25, 50))], [])

        result = second_round(
            {"question": "What happened and why?", "candidates": [document(i) for i in range(50)]},
            decision,
            retrieve,
        )
        rows = [d["corpus_row"] for d in result["candidates"]]
        self.assertEqual(observed, [decision["follow_up_question"]])
        self.assertEqual(len(rows), 50)
        self.assertEqual(len(set(rows)), 50)
        self.assertTrue(set(rows) & set(range(50, 75)))
        self.assertTrue(set(rows) & set(range(25)))
        self.assertTrue(set(rows[:25]) <= set(range(25, 50)))

    def test_inconsistent_decisions_are_rejected_without_fallback(self):
        valid = {
            "needs_more_evidence": True,
            "assessment": "Missing impact.",
            "missing_information": ["Impact"],
            "follow_up_question": "Who was affected?",
        }
        for changes in (
            {"needs_more_evidence": "true"},
            {"missing_information": []},
            {"follow_up_question": None},
            {"needs_more_evidence": False},
            {"follow_up_question": " ".join((f"word{i}" for i in range(129)))},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parse_decision(json.dumps({**valid, **changes}))
        self.assertEqual(parse_decision("```json\n" + json.dumps(valid) + "\n```"), valid)


def doc(row):
    return {
        "corpus_row": row,
        "doc_id": str(row),
        "title": str(row),
        "content": "body",
        "source_type": "test",
        "score": 0.0,
    }


class RefinementTests(unittest.TestCase):
    def test_fusion_can_admit_new_documents_and_keeps_fifty(self):
        original = [doc(i) for i in range(1, 51)]
        expanded = [doc(i) for i in range(51, 101)]
        fused = weighted_fuse(original, expanded, 0.75)
        self.assertEqual(len(fused), 50)
        self.assertEqual(len({d["corpus_row"] for d in fused}), 50)
        self.assertTrue(any((d["corpus_row"] > 50 for d in fused)))
        anchored = weighted_fuse(original, expanded, 0.5)
        self.assertEqual({d["corpus_row"] for d in anchored}, set(range(1, 51)))

    def test_identity_and_duplicate_lists_are_checked(self):
        with self.assertRaises(ValueError):
            weighted_fuse([doc(1), doc(1)], [doc(2)], 1.0)
        changed = {**doc(1), "content": "different"}
        with self.assertRaises(ValueError):
            weighted_fuse([doc(1)], [changed], 1.0)
        for weight in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                weighted_fuse([doc(1)], [doc(2)], weight)

    def test_fusion_does_not_consult_hidden_labels_or_input_scores(self):
        (a, b) = ([doc(1), doc(2)], [doc(2), doc(3)])
        before = [d["corpus_row"] for d in weighted_fuse(a, b, 0.75)]
        for item in a + b:
            item.update(score=999, expected_doc_ids=["secret"])
        self.assertEqual(before, [d["corpus_row"] for d in weighted_fuse(a, b, 0.75)])

    def test_only_count_guidance_changes_between_count_prompts(self):
        r = {
            "question": "Why did the job stop?",
            "retrieved_document_count": 50,
            "term_counts": [{"term": "job", "documents_matched": 3}],
            "content": "SECRET_PASSAGE",
            "expected_doc_ids": ["SECRET_ID"],
        }
        (short, guided) = (messages(r, "short_counts"), messages(r, "guided_counts"))
        self.assertEqual(short[1], guided[1])
        self.assertTrue(guided[0]["content"].startswith(short[0]["content"]))
        self.assertEqual(
            json.loads(messages(r, "short_question")[1]["content"]), {"question": r["question"]}
        )
        for mode in ("short_question", "short_counts", "guided_counts"):
            value = json.dumps(messages(r, mode))
            self.assertNotIn("SECRET_PASSAGE", value)
            self.assertNotIn("SECRET_ID", value)

    def test_fences_empty_and_short_phrase_outputs_are_valid(self):
        self.assertEqual(
            keywords('```json\n{"keywords":["timeout","termination"]}\n```', "job"),
            ["timeout", "termination"],
        )
        self.assertEqual(keywords('{"keywords":[]}', "job"), [])
        self.assertEqual(
            keywords('{"keywords":["server session termination metric"]}', "job"),
            ["server session termination metric"],
        )
        with self.assertRaises(ValueError):
            keywords('{"keywords":["a","b","c","d"]}', "job")


class QueryExpansionTests(unittest.TestCase):
    def test_counts_are_document_frequencies_and_keep_zeroes(self):
        candidates = [
            {"corpus_row": 1, "title": "Access", "content": "network network café"},
            {"corpus_row": 2, "title": "Other", "content": "networking cafe"},
        ]
        counts = match_counts("network ACCESS cafe missing", candidates)
        self.assertEqual(
            {item["term"]: item["documents_matched"] for item in counts},
            {"network": 1, "access": 1, "cafe": 2, "missing": 0},
        )
        self.assertEqual(counts[0]["term"], "cafe")

    def test_duplicate_source_rows_are_rejected(self):
        doc = {"corpus_row": 1, "title": "", "content": "network"}
        with self.assertRaises(ValueError):
            match_counts("network", [doc, doc])

    def test_valid_keywords_and_empty_expansion(self):
        self.assertEqual(
            parse_keywords('{"keywords": ["access revocation"]}', "offboarding"),
            ["access revocation"],
        )
        self.assertEqual(parse_keywords('{"keywords": []}', "offboarding"), [])

    def test_phrase_length_is_a_preference_not_a_hard_query_limit(self):
        raw = '{"keywords": ["server streaming session end metric"]}'
        self.assertEqual(
            parse_keywords(raw, "session termination"), ["server streaming session end metric"]
        )

    def test_complete_json_fence_preserves_keywords_and_rejects_extra_prose(self):
        raw = '{"keywords": ["visual hierarchy", "grid components"]}'
        for wrapped in (raw, "```json\n" + raw + "\n```", "```\n" + raw + "\n```"):
            self.assertEqual(parse_keywords(wrapped, "UI"), ["visual hierarchy", "grid components"])
        for invalid in ("Here is the answer: " + raw, "```json\n" + raw + "\n``` extra"):
            with self.assertRaises(ValueError):
                parse_keywords(invalid, "UI")

    def test_malformed_or_excessive_expansions_are_rejected(self):
        invalid = [
            '{"keywords": [], "answer": "hidden"}',
            '{"keywords": "word"}',
            '{"keywords": [5]}',
            '{"keywords": ["", " "]}',
            '{"keywords": ["ACCESS", "access"]}',
            json.dumps({"keywords": [f"term{i}" for i in range(9)]}),
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises((ValueError, TypeError)):
                parse_keywords(raw, "original question")
        with self.assertRaises(ValueError):
            parse_keywords('{"keywords": ["extra"]}', " ".join((f"term{i}" for i in range(128))))

    def test_query_tokenization_matches_existing_lexical_search(self):
        self.assertEqual(terms("API API v2—rate_limit?"), ["api", "v2", "rate", "limit"])


class PipelineFeedbackTests(unittest.TestCase):
    def setUp(self):
        from enterprise_retrieval.search import Retriever

        self.retriever = Retriever.__new__(Retriever)
        self.retriever.reranker = None
        self.queries = []

        def lexical(question):
            self.queries.append(question)
            return [document(i) for i in range(1, 51)]

        self.retriever.lexical = lexical

    def test_question_control_needs_only_one_augmented_search(self):
        def generate(messages, max_tokens):
            self.assertEqual(json.loads(messages[1]["content"]), {"question": "offboarding"})
            self.assertEqual(max_tokens, 192)
            return '{"keywords":["revocation"]}'

        self.retriever.search("offboarding", "bm25", "question", generate)
        self.assertEqual(self.queries, ["offboarding revocation"])

    def test_counts_use_first_search_and_reranking_uses_original_question(self):
        from unittest.mock import Mock

        self.retriever.reranker = Mock()
        self.retriever.reranker.rank.side_effect = lambda question, docs, key: (docs, {})

        def generate(messages, max_tokens):
            value = json.loads(messages[1]["content"])
            self.assertEqual(value["retrieved_document_count"], 50)
            self.assertIn("original_query_term_document_counts", value)
            return '{"keywords":["revocation"]}'

        result = self.retriever.search("offboarding", "bm25", "counts", generate)
        self.assertEqual(self.queries, ["offboarding", "offboarding revocation"])
        self.assertEqual(self.retriever.reranker.rank.call_args.args[0], "offboarding")
        self.assertEqual(len(result["documents"]), 5)

    def test_dense_and_feedback_configuration_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            self.retriever.search("question", "dense", "counts", lambda *args: "")
        with self.assertRaises(ValueError):
            self.retriever.search("question", "hybrid", "followup", lambda *args: "")

    def test_endpoint_preserves_messages_and_rejects_truncated_generation(self):
        from io import BytesIO
        from unittest.mock import patch

        from enterprise_retrieval.feedback import ChatEndpoint

        observed = []

        def respond(request, timeout):
            observed.append(json.loads(request.data))
            return BytesIO(
                json.dumps(
                    {
                        "choices": [
                            {"finish_reason": "stop", "message": {"content": '{"keywords":[]}'}}
                        ]
                    }
                ).encode()
            )

        endpoint = ChatEndpoint("http://localhost:8000/v1/chat/completions", "test-model")
        messages = [{"role": "user", "content": "search terms"}]
        with patch("urllib.request.urlopen", side_effect=respond):
            self.assertEqual(endpoint(messages, 192), '{"keywords":[]}')
        self.assertEqual(observed[0]["messages"], messages)
        self.assertEqual(observed[0]["max_tokens"], 192)
        with patch(
            "urllib.request.urlopen",
            return_value=BytesIO(json.dumps({"choices": [{"finish_reason": "length"}]}).encode()),
        ):
            with self.assertRaises(ValueError):
                endpoint(messages, 192)
