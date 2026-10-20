"""Two first-search feedback approaches, with reference labels excluded from prompts."""

import json
import math
import re
import sqlite3
import urllib.request
from pathlib import Path

PROMPTS = Path(__file__).parent / "prompts"

SHORT_SYSTEM = (PROMPTS / "expansion.txt").read_text()

MAX_KEYWORDS = 8


def terms(text):
    """Match the existing BM25 query preprocessor, without a new stop-word filter."""
    return list(dict.fromkeys(re.findall(r"[^\W_]+", text.lower())))


def match_counts(question, candidates):
    """Count unique source rows using the index's default FTS5 unicode61 tokenizer."""
    if len({d["corpus_row"] for d in candidates}) != len(candidates):
        raise ValueError("Feedback documents must have distinct source rows")
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE VIRTUAL TABLE evidence USING fts5(text)")
        db.executemany(
            "INSERT INTO evidence(text) VALUES (?)",
            [(d["title"] + "\n" + d["content"],) for d in candidates],
        )
        counts = [
            {
                "term": term,
                "documents_matched": db.execute(
                    "SELECT count(*) FROM evidence WHERE evidence MATCH ?", (f'"{term}"',)
                ).fetchone()[0],
            }
            for term in terms(question)
        ]
    return sorted(counts, key=lambda c: -c["documents_matched"])


def parse_keywords(raw, question):
    # A complete JSON code block is a transport wrapper, not a change to its content.
    fenced = re.fullmatch(r"\s*```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```\s*", raw, re.I)
    value = json.loads(fenced.group(1) if fenced else raw)
    if not isinstance(value, dict) or set(value) != {"keywords"}:
        raise ValueError("Expected a JSON object containing only keywords")
    keywords = value["keywords"]
    if not isinstance(keywords, list) or len(keywords) > MAX_KEYWORDS:
        raise ValueError("Expected at most eight keywords")
    seen = set()
    for keyword in keywords:
        if not isinstance(keyword, str) or not terms(keyword):
            raise ValueError("Each keyword must be a nonempty searchable string")
        normalized = " ".join(terms(keyword))
        if normalized in seen:
            raise ValueError("Repeated expansion keyword")
        seen.add(normalized)
    expanded = question + (" " + " ".join(keywords) if keywords else "")
    if not 1 <= len(terms(expanded)) <= 128:
        raise ValueError("Expanded query exceeds the existing BM25 term limit")
    return keywords


SHORT_MODES = ("short_question", "short_counts", "guided_counts")

COUNT_GUIDANCE = """
The counts report how many of the original top50 retrieved documents match each original
query term; they are not corpus-wide frequencies or relevance labels. Ignore function words.
A rare person, organization or product name may precisely identify the correct document;
do not replace it or assume that its low count is a problem. A high count does not establish
importance. Consider alternative wording for central concepts when the counts and question
together suggest it might help. Do not expand every low-count term or broaden the subject."""


def expansion_messages(record, mode):
    if mode not in SHORT_MODES:
        raise ValueError(f"Unknown generation mode: {mode}")
    payload = {"question": record["question"]}
    if mode != "short_question":
        payload.update(
            retrieved_document_count=record["retrieved_document_count"],
            original_query_term_document_counts=record["term_counts"],
        )
    return [
        {
            "role": "system",
            "content": SHORT_SYSTEM + (COUNT_GUIDANCE if mode == "guided_counts" else ""),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def keywords(raw, question):
    result = parse_keywords(raw, question)
    if len(result) > 3:
        raise ValueError("Refinement requires at most three keyword entries")
    return result


def weighted_fuse(original, expanded, expansion_weight, limit=50):
    """Apply weighted RRF60 to source rows, without reference labels or scores."""
    if not math.isfinite(expansion_weight) or expansion_weight <= 0:
        raise ValueError("Expansion weight must be finite and positive")
    merged = {}
    for name, documents, weight in (
        ("original", original, 1.0),
        ("expansion", expanded, expansion_weight),
    ):
        seen = set()
        for rank, document in enumerate(documents, 1):
            row = document["corpus_row"]
            if row in seen:
                raise ValueError("Repeated source row in a retrieval list")
            seen.add(row)
            if row not in merged:
                merged[row] = {**document, "score": 0.0}
            entry = merged[row]
            if any(entry[k] != document[k] for k in ("doc_id", "source_type", "title", "content")):
                raise ValueError("Source row identity has conflicting contents")
            entry["score"] += weight / (60 + rank)
            entry[name + "_rank"] = rank
    return sorted(merged.values(), key=lambda d: (-d["score"], d["corpus_row"]))[:limit]


class ChatEndpoint:
    """Call a chat-completions endpoint; model hosting is configured separately."""

    def __init__(self, url, model, api_key=None):
        self.url = url
        self.model = model
        self.api_key = api_key

    def __call__(self, messages, max_tokens):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            choice = json.load(response)["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError("Feedback generation did not finish within the token budget")
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("The endpoint must return text content")
        return content
