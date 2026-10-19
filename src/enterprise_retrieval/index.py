"""Build shared 510-token passages and the passage BM25 index."""

import json
import sqlite3
import time
from bisect import bisect_right
from collections import Counter
from pathlib import Path

from .corpus import sha256, write_json

MODEL_ID = "BAAI/bge-base-en-v1.5"
MODEL_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
MAX_TOKENS = 512
MAX_SOURCE_CHARS = 4500


def prepare_tokenizer(destination):
    from huggingface_hub import snapshot_download

    if destination.exists():
        raise FileExistsError(destination)
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=destination,
        allow_patterns=[
            "config.json",
            "tokenizer*.json",
            "special_tokens_map.json",
            "vocab.txt",
            "README.md",
        ],
    )
    write_json(
        destination / "snapshot.json",
        {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": "MIT",
            "source": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
        },
    )


PAYLOAD_TOKENS = MAX_TOKENS - 2


def token_windows(ids, offsets, text_length):
    """Cover every token once and partition the original characters without gaps."""
    if len(ids) != len(offsets):
        raise ValueError("Token IDs and offsets differ")
    if not ids:
        yield 0, 0, 0, text_length, []
        return
    for start in range(0, len(ids), PAYLOAD_TOKENS):
        end = min(start + PAYLOAD_TOKENS, len(ids))
        char_start = offsets[start][0] if start else 0
        char_end = offsets[end][0] if end < len(ids) else text_length
        yield start, end, char_start, char_end, ids[start:end]


def prefix_coverage(offsets, prefix_offsets, title_length, content_length):
    visible_end = max((end for _, end in prefix_offsets[:PAYLOAD_TOKENS]), default=0)
    retained = bisect_right([end for _, end in offsets], visible_end)
    return {
        "full_payload_tokens": len(offsets),
        "old_visible_full_tokens": retained,
        "full_characters": title_length + 1 + content_length,
        "old_visible_characters": visible_end,
        "content_characters": content_length,
        "old_visible_content_characters": max(
            0, min(content_length, visible_end - title_length - 1)
        ),
    }


def prepare(data_dir, model_dir, corpus_index, output):
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer

    if output.exists():
        raise FileExistsError(output)
    dataset = json.loads((data_dir / "dataset.json").read_text())
    source = dataset["files"]["documents"]
    documents = data_dir / source["path"]
    if sha256(documents) != source["sha256"]:
        raise ValueError("Corpus fingerprint differs")
    snapshot = json.loads((model_dir / "snapshot.json").read_text())
    if snapshot["model_id"] != MODEL_ID or snapshot["revision"] != MODEL_REVISION:
        raise ValueError("Unexpected embedding tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if not tokenizer.is_fast or tokenizer.num_special_tokens_to_add() != 2:
        raise ValueError("This partition requires the pinned BERT fast tokenizer")
    with sqlite3.connect(f"{corpus_index.resolve().as_uri()}?mode=ro", uri=True) as corpus:
        metadata = json.loads(corpus.execute("SELECT value FROM metadata").fetchone()[0])
    if metadata["dataset_revision"] != dataset["revision"]:
        raise ValueError("Corpus revisions differ")
    output.mkdir(parents=True)
    manifest = {
        "model": snapshot,
        "documents": source,
        "dataset_revision": dataset["revision"],
        "corpus": metadata,
        "max_tokens": MAX_TOKENS,
        "payload_tokens": PAYLOAD_TOKENS,
        "overlap": 0,
        "input": "Full title + newline + full content; no character cutoff",
        "title_repeated": False,
        "special_tokens": 2,
        "code_sha256": {
            name: sha256(Path(__file__).parent / name) for name in ("index.py", "corpus.py")
        },
    }
    write_json(output / "manifest.json", manifest)
    schema = arrow.schema(
        [
            ("chunk_row", arrow.int64()),
            ("corpus_row", arrow.int64()),
            ("start_token", arrow.int32()),
            ("end_token", arrow.int32()),
            ("char_start", arrow.int32()),
            ("char_end", arrow.int32()),
            ("input_ids", arrow.list_(arrow.int32())),
        ]
    )
    totals, by_source = Counter(), {}
    count = chunks = 0
    started = time.monotonic()
    write_json(output / "status.json", {"state": "running", "documents": 0, "chunks": 0})
    try:
        with (
            sqlite3.connect(output / "bm25.sqlite") as db,
            parquet.ParquetWriter(output / "chunks.parquet", schema, compression="zstd") as writer,
        ):
            db.execute("CREATE VIRTUAL TABLE chunks USING fts5(corpus_row UNINDEXED, text)")
            for batch in parquet.ParquetFile(documents).iter_batches(batch_size=256):
                records = batch.to_pylist()
                texts = [d["title"] + "\n" + d["content"] for d in records]
                prefixes = [d["title"] + "\n" + d["content"][:MAX_SOURCE_CHARS] for d in records]
                full = tokenizer(
                    texts,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    truncation=False,
                    verbose=False,
                )
                old = tokenizer(
                    prefixes,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    truncation=False,
                    verbose=False,
                )
                rows, lexical = [], []
                for doc, text, ids, offsets, prefix_offsets in zip(
                    records, texts, full["input_ids"], full["offset_mapping"], old["offset_mapping"]
                ):
                    count += 1
                    coverage = prefix_coverage(
                        offsets, prefix_offsets, len(doc["title"]), len(doc["content"])
                    )
                    coverage.update(
                        documents=1,
                        token_truncated_documents=int(len(prefix_offsets) > PAYLOAD_TOKENS),
                        character_truncated_documents=int(len(doc["content"]) > MAX_SOURCE_CHARS),
                    )
                    totals.update(coverage)
                    by_source.setdefault(doc["source_type"], Counter()).update(coverage)
                    emitted = 0
                    for a, b, c, d, tokens in token_windows(ids, offsets, len(text)):
                        chunks += 1
                        emitted += len(tokens)
                        rows.append(
                            dict(
                                chunk_row=chunks,
                                corpus_row=count,
                                start_token=a,
                                end_token=b,
                                char_start=c,
                                char_end=d,
                                input_ids=tokens,
                            )
                        )
                        lexical.append((chunks, count, text[c:d]))
                    if emitted != len(ids):
                        raise ValueError("Chunking lost or repeated tokens")
                db.executemany("INSERT INTO chunks(rowid,corpus_row,text) VALUES(?,?,?)", lexical)
                writer.write_table(arrow.Table.from_pylist(rows, schema=schema))
                if count % 12800 == 0 or count == source["rows"]:
                    status = {
                        "state": "running",
                        "documents": count,
                        "chunks": chunks,
                        "seconds": round(time.monotonic() - started, 2),
                    }
                    write_json(output / "status.json", status)
                    print(json.dumps(status), flush=True)
            if count != source["rows"]:
                raise ValueError("Incomplete document coverage")
            write_json(
                output / "coverage.json",
                {
                    "totals": dict(totals),
                    "by_source": {k: dict(v) for k, v in by_source.items()},
                    "old_omitted_token_fraction": 1
                    - totals["old_visible_full_tokens"] / totals["full_payload_tokens"],
                    "old_omitted_character_fraction": 1
                    - totals["old_visible_characters"] / totals["full_characters"],
                    "old_omitted_content_character_fraction": 1
                    - totals["old_visible_content_characters"] / totals["content_characters"],
                    "definition": "Token counts exclude CLS/SEP; a full-document token is visible "
                    "when its end offset is within the old encoder's visible character prefix.",
                    "chunked_token_coverage": 1.0,
                    "chunks": chunks,
                },
            )
            print("Coverage measured; optimizing lexical index", flush=True)
            db.execute("INSERT INTO chunks(chunks) VALUES('optimize')")
        manifest.update(
            chunks=chunks,
            documents_count=count,
            chunks_sha256=sha256(output / "chunks.parquet"),
            lexical_sha256=sha256(output / "bm25.sqlite"),
        )
        write_json(output / "manifest.json", manifest)
        write_json(output / "status.json", {**status, "state": "completed"})
    except BaseException as error:
        write_json(
            output / "status.json",
            {"state": "failed", "documents": count, "chunks": chunks, "error": repr(error)},
        )
        raise
