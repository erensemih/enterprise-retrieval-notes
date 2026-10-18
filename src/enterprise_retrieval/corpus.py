"""Download the pinned benchmark and retain source-row identity in SQLite."""

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

DATASET_ID = "onyx-dot-app/EnterpriseRAG-Bench"


DATASET_REVISION = "69916e31c68aa5963c00248fd7f0bc12d04fd235"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(data_dir):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    data_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("documents", "questions"):
        path = Path(
            hf_hub_download(
                DATASET_ID,
                f"data/{name}/test.parquet",
                repo_type="dataset",
                revision=DATASET_REVISION,
                local_dir=data_dir,
            )
        )
        parquet = pq.ParquetFile(path)
        files[name] = {
            "path": str(path.relative_to(data_dir)),
            "rows": parquet.metadata.num_rows,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "schema": str(parquet.schema_arrow),
        }
        print(name, json.dumps(files[name]), flush=True)
    questions = pq.read_table(data_dir / files["questions"]["path"]).to_pylist()
    write_json(data_dir / "questions.json", questions)
    write_json(
        data_dir / "dataset.json",
        {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "files": files,
            "question_types": dict(Counter(q["question_type"] for q in questions)),
        },
    )


def build_index(data_dir, index_path):
    import pyarrow.parquet as pq

    manifest = json.loads((data_dir / "dataset.json").read_text())
    source = data_dir / manifest["files"]["documents"]["path"]
    if sha256(source) != manifest["files"]["documents"]["sha256"]:
        raise ValueError("Document parquet checksum differs from dataset.json")
    if index_path.exists():
        raise FileExistsError(f"Choose a new index path: {index_path}")
    started = time.monotonic()
    counts = Counter()
    seen = {}
    duplicates = []
    row_count = 0
    with sqlite3.connect(index_path) as db:
        db.execute(
            "CREATE VIRTUAL TABLE documents USING fts5("
            "doc_id UNINDEXED, source_type UNINDEXED, title, content)"
        )
        for batch in pq.ParquetFile(source).iter_batches(batch_size=1000):
            rows = []
            for doc in batch.to_pylist():
                if not all(
                    isinstance(doc.get(k), str)
                    for k in ("doc_id", "source_type", "title", "content")
                ):
                    raise ValueError(f"Invalid document fields: {doc.get('doc_id')}")
                row_count += 1
                fingerprint = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
                if doc["doc_id"] in seen:
                    first_row, first_hash = seen[doc["doc_id"]]
                    duplicates.append(
                        {
                            "doc_id": doc["doc_id"],
                            "first_row": first_row,
                            "duplicate_row": row_count,
                            "same_payload": fingerprint == first_hash,
                        }
                    )
                else:
                    seen[doc["doc_id"]] = (row_count, fingerprint)
                counts[doc["source_type"]] += 1
                rows.append(tuple(doc[k] for k in ("doc_id", "source_type", "title", "content")))
            db.executemany("INSERT INTO documents VALUES (?, ?, ?, ?)", rows)
            if row_count % 50000 == 0:
                print(f"Indexed {row_count} records", flush=True)
        if row_count != manifest["files"]["documents"]["rows"]:
            raise ValueError("Indexed row count differs from dataset manifest")
        db.execute("INSERT INTO documents(documents) VALUES ('optimize')")
        db.execute("CREATE TABLE metadata (value TEXT NOT NULL)")
        metadata = {
            "dataset_revision": manifest["revision"],
            "document_count": row_count,
            "distinct_document_ids": len(seen),
            "duplicate_records": duplicates,
            "record_identity": "one-based parquet row; duplicate IDs retained without changes",
            "source_counts": dict(counts),
            "retriever": "SQLite FTS5 BM25",
            "unit": "whole document",
            "fields": ["title", "content"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        db.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
    write_json(index_path.with_suffix(".json"), metadata)
    print(json.dumps(metadata), flush=True)


def search(index_path, question, top_k=5):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    terms = list(dict.fromkeys(re.findall(r"[^\W_]+", question.lower())))
    if not terms:
        return []
    if len(terms) > 128:
        raise ValueError("The BM25 baseline accepts at most 128 distinct query terms")
    expression = " OR ".join(f'"{term}"' for term in terms)
    with sqlite3.connect(f"{Path(index_path).resolve().as_uri()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            "SELECT rowid AS corpus_row, doc_id, source_type, title, content, "
            "bm25(documents) AS score "
            "FROM documents WHERE documents MATCH ? ORDER BY score, rowid LIMIT ?",
            (expression, top_k),
        ).fetchall()
    return [dict(row) for row in rows]
