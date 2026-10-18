"""Reference-document recall and grouped paired comparisons."""

import sqlite3
from statistics import mean


def retrieval_metrics(expected_ids, retrieved_ids, ambiguous_ids, question_type, k):
    expected = set(expected_ids or [])
    if question_type in {"high_level", "info_not_found"}:
        reason = "no_document_reference_for_this_task_type"
    elif not expected:
        reason = "missing_document_reference"
    elif expected & ambiguous_ids:
        reason = "ambiguous_reference_document_id"
    else:
        ranked = retrieved_ids[:k]
        matched = expected & set(ranked)
        first_rank = next((i for i, doc in enumerate(ranked, 1) if doc in expected), None)
        return {
            "status": "scored",
            "expected_count": len(expected),
            "matched_count": len(matched),
            "recall": len(matched) / len(expected),
            "reciprocal_rank": 1 / first_rank if first_rank else 0,
        }
    return {"status": reason, "recall": None, "reciprocal_rank": None}


def paired_interval(differences, split, repeats=2000):
    """Resample shared-reference groups, retaining a question-weighted mean difference."""
    import numpy as np

    groups = [
        [differences[q] for q in group["question_ids"] if q in differences]
        for group in split["groups"]
    ]
    groups = [g for g in groups if g]
    if not groups or sum(map(len, groups)) != len(differences):
        raise ValueError("Paired tasks do not match split groups")
    totals = np.array([sum(g) for g in groups])
    counts = np.array([len(g) for g in groups])
    sampled = np.random.default_rng(0).integers(0, len(groups), (repeats, len(groups)))
    means = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "difference": sum(differences.values()) / len(differences),
        "interval_95": np.quantile(means, [0.025, 0.975]).tolist(),
        "bootstrap_sem": float(means.std(ddof=1)),
        "groups": len(groups),
        "questions": len(differences),
        "repeats": repeats,
        "seed": 0,
    }


def evaluate(retriever, questions, method="bm25", feedback="none", generate=None):
    """Retrieve before consulting references; missing and ambiguous labels remain NA."""
    import json

    with sqlite3.connect(retriever.corpus_index) as db:
        metadata = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
    ambiguous = {row["doc_id"] for row in metadata.get("duplicate_records", [])}
    ids = [q["question_id"] for q in questions]
    if len(ids) != len(set(ids)) or not questions:
        raise ValueError("Expected a nonempty set of distinct question IDs")
    cases = []
    for question in questions:
        result = retriever.search(question["question"], method, feedback, generate)
        documents = result["documents"]
        metric = retrieval_metrics(
            question.get("expected_doc_ids"),
            [doc["doc_id"] for doc in documents],
            ambiguous,
            question["question_type"],
            5,
        )
        cases.append({"question_id": question["question_id"], **metric})
    scored = [row for row in cases if row["status"] == "scored"]
    return {
        "questions": len(cases),
        "scored": len(scored),
        "unscored": len(cases) - len(scored),
        "reference_recall_at_5": mean(row["recall"] for row in scored) if scored else None,
        "cases": cases,
    }
