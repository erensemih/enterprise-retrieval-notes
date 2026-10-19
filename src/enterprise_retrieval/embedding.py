"""Embed the original passage character spans with pinned Qwen3-Embedding-0.6B."""

import hashlib
import importlib.metadata
import json
import time
from itertools import groupby, islice
from pathlib import Path

from .corpus import sha256, write_json

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"


MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


DIMENSIONS = 1024


MAX_TOKENS = 32768


QUERY_PREFIX = (
    "Instruct: Given a question about company information, retrieve relevant passages "
    "that answer the question\nQuery:"
)


BATCH_SIZE = 64


def prepare(destination):
    from huggingface_hub import snapshot_download

    if destination.exists():
        raise FileExistsError(destination)
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=destination,
        allow_patterns=[
            "model.safetensors",
            "config.json",
            "tokenizer*.json",
            "vocab.json",
            "merges.txt",
            "README.md",
            "1_Pooling/config.json",
        ],
    )
    write_json(
        destination / "snapshot.json",
        {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": "Apache-2.0",
            "source": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
        },
    )


def last_token_pool(hidden, attention_mask):
    """Select the last real token for either left- or right-padded batches."""
    import torch

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last = (
        positions.expand_as(attention_mask).masked_fill(attention_mask == 0, -1).max(dim=1).values
    )
    if (last < 0).any():
        raise ValueError("An embedding input has no tokens")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class QwenEncoder:
    query_prefix = QUERY_PREFIX
    max_tokens = MAX_TOKENS

    def __init__(self, model_dir):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available():
            raise ValueError("Encoding requires a GPU Slurm job")
        snapshot = json.loads((model_dir / "snapshot.json").read_text())
        if snapshot["model_id"] != MODEL_ID or snapshot["revision"] != MODEL_REVISION:
            raise ValueError("Unexpected Qwen snapshot")
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, padding_side="left"
        )
        self.model = (
            AutoModel.from_pretrained(
                model_dir, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
            )
            .to("cuda")
            .eval()
        )
        if self.model.config.hidden_size != DIMENSIONS:
            raise ValueError("Unexpected embedding dimensions")

    def encode(self, texts):
        import torch
        import torch.nn.functional as functional

        tokens = self.tokenizer(texts, truncation=False, padding=False)["input_ids"]
        lengths = [len(ids) for ids in tokens]
        if not lengths or min(lengths) == 0 or max(lengths) > MAX_TOKENS:
            raise ValueError("Invalid token count; silent truncation is forbidden")
        inputs = self.tokenizer.pad({"input_ids": tokens}, padding=True, return_tensors="pt").to(
            "cuda"
        )
        with torch.inference_mode():
            hidden = self.model(**inputs, use_cache=False).last_hidden_state
            vectors = last_token_pool(hidden, inputs["attention_mask"]).float()
            if not torch.isfinite(vectors).all() or (vectors.norm(dim=1) == 0).any():
                raise ValueError("Invalid embedding")
            vectors = functional.normalize(vectors, p=2, dim=1)
        return vectors, lengths


def source_windows(documents, chunks):
    """Join ordered chunk offsets to original text, checking a lossless character partition."""
    groups = iter(groupby(chunks, key=lambda c: c["corpus_row"]))
    chunk_row = 0
    for row, document in enumerate(documents, 1):
        entry = next(groups, None)
        if entry is None:
            raise ValueError("A source document has no chunks")
        parent, group = entry
        if parent != row:
            raise ValueError("Missing or reordered source document")
        text = document["title"] + "\n" + document["content"]
        end = 0
        for chunk in group:
            chunk_row += 1
            if (
                chunk["chunk_row"] != chunk_row
                or chunk["char_start"] != end
                or not end <= chunk["char_end"] <= len(text)
            ):
                raise ValueError("Invalid source chunk partition")
            end = chunk["char_end"]
            yield chunk_row, row, text[chunk["char_start"] : end]
        if end != len(text):
            raise ValueError("Source document was truncated")
    if next(groups, None) is not None:
        raise ValueError("Chunks have an unknown source document")


def parquet_rows(path, columns=None):
    import pyarrow.parquet as parquet

    for batch in parquet.ParquetFile(path).iter_batches(batch_size=4096, columns=columns):
        yield from batch.to_pylist()


def encode(chunks_dir, data_dir, model_dir, output):
    import numpy as np

    if output.exists():
        raise FileExistsError(output)
    cm = json.loads((chunks_dir / "manifest.json").read_text())
    if json.loads((chunks_dir / "status.json").read_text())["state"] != "completed":
        raise ValueError("Chunk preparation is incomplete")
    documents = data_dir / cm["documents"]["path"]
    if sha256(documents) != cm["documents"]["sha256"]:
        raise ValueError("Original corpus fingerprint differs")
    if sha256(chunks_dir / "chunks.parquet") != cm["chunks_sha256"]:
        raise ValueError("Chunk fingerprint differs")
    output.mkdir(parents=True)
    manifest = {
        "model": json.loads((model_dir / "snapshot.json").read_text()),
        "chunks": cm["chunks"],
        "documents": cm["documents_count"],
        "dimensions": DIMENSIONS,
        "chunk_manifest_sha256": sha256(chunks_dir / "manifest.json"),
        "input": "Original title+newline+content at the existing chunk character offsets",
        "pooling": "Last real token, FP32 L2 normalization",
        "query_prefix": QUERY_PREFIX,
        "dtype": "BF16 model, FP32 vectors",
        "max_input_tokens": MAX_TOKENS,
        "batch_size": BATCH_SIZE,
        "truncation": False,
        "code_sha256": {
            name: sha256(Path(__file__).parent / name)
            for name in (
                "embedding.py",
                "index.py",
                "search.py",
                "corpus.py",
            )
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "pyarrow")
        },
    }
    write_json(output / "manifest.json", manifest)
    matrix = np.lib.format.open_memmap(
        output / "vectors.npy", mode="w+", dtype="float32", shape=(cm["chunks"], DIMENSIONS)
    )
    parents = np.lib.format.open_memmap(
        output / "parents.npy", mode="w+", dtype="int64", shape=(cm["chunks"],)
    )
    count = characters = max_tokens = token_count = 0
    digest = hashlib.sha256()
    started = time.monotonic()
    write_json(output / "status.json", {"state": "running", "chunks": 0})
    try:
        encoder = QwenEncoder(model_dir)
        import torch

        examples = [
            QUERY_PREFIX + "What is the capital of China?",
            "The capital of China is Beijing.",
            "Gravity is a force that attracts objects toward one another.",
        ]
        together, _ = encoder.encode(examples)
        separate = torch.cat([encoder.encode([text])[0] for text in examples])
        agreement = (together * separate).sum(dim=1)
        similarities = (together[:1] @ together[1:].T).tolist()[0]
        if min(agreement.tolist()) < 0.999 or similarities[0] <= similarities[1]:
            raise ValueError("Pooling/padding/relevance smoke check failed")
        manifest["smoke_check"] = {
            "batch_single_cosine": agreement.tolist(),
            "relevant_unrelated_scores": similarities,
        }
        write_json(output / "manifest.json", manifest)
        windows = source_windows(
            parquet_rows(documents),
            parquet_rows(
                chunks_dir / "chunks.parquet", ["chunk_row", "corpus_row", "char_start", "char_end"]
            ),
        )
        while records := list(islice(windows, BATCH_SIZE)):
            vectors, lengths = encoder.encode([r[2] for r in records])
            matrix[count : count + len(records)] = vectors.cpu().numpy()
            parents[count : count + len(records)] = [r[1] for r in records]
            for chunk_row, parent, text in records:
                encoded = text.encode("utf-8")
                digest.update(f"{chunk_row}:{parent}:{len(encoded)}:".encode())
                digest.update(encoded)
                characters += len(text)
            count += len(records)
            max_tokens = max(max_tokens, max(lengths))
            token_count += sum(lengths)
            if count % (BATCH_SIZE * 100) == 0 or count == cm["chunks"]:
                status = {
                    "state": "running",
                    "chunks": count,
                    "expected_chunks": cm["chunks"],
                    "seconds": round(time.monotonic() - started, 2),
                    "max_input_tokens": max_tokens,
                }
                write_json(output / "status.json", status)
                print(json.dumps(status), flush=True)
        coverage = json.loads((chunks_dir / "coverage.json").read_text())
        if count != cm["chunks"] or characters != coverage["totals"]["full_characters"]:
            raise ValueError("Incomplete source coverage")
        counts = np.bincount(parents - 1, minlength=cm["documents_count"])
        if len(counts) != cm["documents_count"] or np.any(counts == 0):
            raise ValueError("A source document is missing")
        np.save(output / "document_spans.npy", np.concatenate(([0], np.cumsum(counts))))
        matrix.flush()
        parents.flush()
        manifest.update(
            {
                "encoded_characters": characters,
                "encoded_tokens": token_count,
                "observed_max_tokens": max_tokens,
                "source_texts_sha256": digest.hexdigest(),
                "files": {
                    name: sha256(output / name)
                    for name in ("vectors.npy", "parents.npy", "document_spans.npy")
                },
            }
        )
        write_json(output / "manifest.json", manifest)
        write_json(output / "status.json", {**status, "state": "completed"})
    except BaseException as error:
        write_json(
            output / "status.json", {"state": "failed", "chunks": count, "error": repr(error)}
        )
        raise
