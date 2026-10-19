"""BM25, exact dense search and reciprocal-rank fusion over source rows."""

import json
import re
import sqlite3
from pathlib import Path

from .embedding import QwenEncoder
from .rerank import Reranker


def load_records(db, rows, scores):
    placeholders = ",".join("?" for _ in rows)
    records = db.execute(
        "SELECT rowid,doc_id,source_type,title,content FROM documents "
        f"WHERE rowid IN ({placeholders})",
        rows,
    ).fetchall()
    by_row = {
        r[0]: dict(zip(("corpus_row", "doc_id", "source_type", "title", "content"), r))
        for r in records
    }
    if len(by_row) != len(rows):
        raise ValueError("Dense row missing from lexical corpus")
    return [{**by_row[row], "score": float(score)} for row, score in zip(rows, scores)]


def best_documents(hits, limit=50):
    selected, seen = [], set()
    for hit in hits:
        if hit["corpus_row"] not in seen:
            selected.append(hit)
            seen.add(hit["corpus_row"])
            if len(selected) == limit:
                break
    return selected


def bm25_search(chunks_index, corpus_index, question):
    terms = list(dict.fromkeys(re.findall(r"[^\W_]+", question.lower())))
    if not terms or len(terms) > 128:
        raise ValueError("Expected 1–128 distinct query terms")
    expression = " OR ".join(f'"{term}"' for term in terms)
    with sqlite3.connect(f"{chunks_index.resolve().as_uri()}?mode=ro", uri=True) as db:
        amount = 256
        while True:
            rows = db.execute(
                "SELECT rowid,corpus_row,bm25(chunks) AS score FROM chunks "
                "WHERE chunks MATCH ? ORDER BY score,rowid LIMIT ?",
                (expression, amount),
            ).fetchall()
            hits = [{"chunk_row": r[0], "corpus_row": int(r[1]), "score": r[2]} for r in rows]
            selected = best_documents(hits)
            if len(selected) == 50 or len(rows) < amount:
                break
            amount *= 2
    with sqlite3.connect(f"{corpus_index.resolve().as_uri()}?mode=ro", uri=True) as corpus:
        candidates = load_records(
            corpus, [h["corpus_row"] for h in selected], [h["score"] for h in selected]
        )
    return candidates, [{k: h[k] for k in ("corpus_row", "chunk_row")} for h in selected]


def document_maxima(scores, parents, documents):
    """Reduce scores over all chunks, rather than over a fixed top-chunk shortlist."""
    import torch

    result = torch.full(
        (scores.shape[0], documents), -torch.inf, dtype=scores.dtype, device=scores.device
    )
    return result.scatter_reduce_(
        1, parents.expand(scores.shape[0], -1), scores, reduce="amax", include_self=True
    )


class Retriever:
    """Load only the models required by the requested retrieval configuration."""

    def __init__(
        self, corpus_index, chunks_dir, dense_index=None, embedding_model=None, reranker_model=None
    ):
        self.corpus_index = Path(corpus_index)
        self.chunks_dir = Path(chunks_dir)
        if not self.corpus_index.is_file() or not (self.chunks_dir / "bm25.sqlite").is_file():
            raise FileNotFoundError("Build the document store and passage index first")
        self.encoder = None
        self.reranker = None
        if dense_index is not None:
            import numpy as np
            import torch

            if embedding_model is None:
                raise ValueError("Dense retrieval requires an embedding model")
            dense_index = Path(dense_index)
            status = json.loads((dense_index / "status.json").read_text())
            manifest = json.loads((dense_index / "manifest.json").read_text())
            if status["state"] != "completed":
                raise ValueError("Dense index is incomplete")
            self.documents = manifest["documents"]
            self.matrix = torch.from_numpy(np.load(dense_index / "vectors.npy")).to("cuda")
            self.parents = torch.from_numpy(np.load(dense_index / "parents.npy") - 1).to("cuda")
            if tuple(self.matrix.shape) != (manifest["chunks"], manifest["dimensions"]):
                raise ValueError("Dense index dimensions differ from its manifest")
            self.encoder = QwenEncoder(Path(embedding_model))
        if reranker_model is not None:
            self.reranker = Reranker(Path(reranker_model))

    def lexical(self, question):
        return bm25_search(self.chunks_dir / "bm25.sqlite", self.corpus_index, question)[0]

    def dense(self, question):
        import torch

        if self.encoder is None:
            raise ValueError("No dense model was loaded")
        (vectors, _) = self.encoder.encode([self.encoder.query_prefix + question])
        with torch.inference_mode():
            scores = vectors @ self.matrix.T
            reduced = document_maxima(scores, self.parents, self.documents)
            positions = torch.argsort(reduced, dim=1, descending=True, stable=True)[:, :50]
            values = reduced.gather(1, positions).cpu().tolist()[0]
            rows = [row + 1 for row in positions.cpu().tolist()[0]]
        with sqlite3.connect(f"{self.corpus_index.resolve().as_uri()}?mode=ro", uri=True) as db:
            return load_records(db, rows, values)

    def search(self, question, method="bm25", feedback="none", generate=None, top_k=5):
        if method not in {"bm25", "dense"} or feedback != "none":
            raise ValueError("Unsupported retrieval configuration")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between one and fifty")
        candidates = self.dense(question) if method == "dense" else self.lexical(question)
        if self.reranker is not None:
            (candidates, _) = self.reranker.rank(question, candidates, "candidate_rank")
        return {"documents": candidates[:top_k], "feedback": {}}
