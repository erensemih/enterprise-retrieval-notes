"""Score the original question and candidate document prefixes with a cross-encoder."""

import json
import math

from .corpus import write_json

MODEL_ID = "BAAI/bge-reranker-v2-m3"


MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


MAX_SOURCE_CHARS = 4500


MAX_TOKENS = 2048


def prepare(destination):
    from huggingface_hub import snapshot_download

    if destination.exists():
        raise FileExistsError("Choose a new model directory")
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=destination,
        allow_patterns=["*.json", "*.safetensors", "sentencepiece.bpe.model", "README.md"],
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


def order_candidates(candidates, scores, rank_key="bm25_rank"):
    if len(candidates) != len(scores) or not all(math.isfinite(s) for s in scores):
        raise ValueError("Each candidate needs a finite reranker score")
    return [
        {**candidates[i], "reranker_score": scores[i], rank_key: i + 1}
        for i in sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    ]


class Reranker:
    def __init__(self, model_dir):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not torch.cuda.is_available():
            raise ValueError("Reranking requires a CUDA device")
        snapshot = json.loads((model_dir / "snapshot.json").read_text())
        if snapshot["revision"] != MODEL_REVISION or snapshot["model_id"] != MODEL_ID:
            raise ValueError("Unexpected reranker snapshot")
        torch.manual_seed(0)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                dtype=torch.bfloat16,
                local_files_only=True,
            )
            .to("cuda")
            .eval()
        )

    def rank(self, question, candidates, rank_key="bm25_rank"):
        import torch

        scores, lengths = [], []
        for start in range(0, len(candidates), 4):
            batch = candidates[start : start + 4]
            pairs = [
                [question, doc["title"] + "\n" + doc["content"][:MAX_SOURCE_CHARS]] for doc in batch
            ]
            lengths.extend(len(ids) for ids in self.tokenizer(pairs)["input_ids"])
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation="only_second",
                max_length=MAX_TOKENS,
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                scores.extend(self.model(**inputs).logits.flatten().float().cpu().tolist())
        ranked = order_candidates(candidates, scores, rank_key)
        return ranked, {
            "untruncated_input_tokens": lengths,
            "token_truncated_count": sum(n > MAX_TOKENS for n in lengths),
        }
