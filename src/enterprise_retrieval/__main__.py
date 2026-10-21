"""Command-line entry points for indexing, retrieval and evaluation."""

import argparse
import json
import os
from pathlib import Path

from . import corpus, embedding, index, rerank
from .evaluation import evaluate
from .feedback import ChatEndpoint
from .search import Retriever


def load_retriever(args):
    work = args.work_dir
    dense = args.method != "bm25"
    return Retriever(
        work / "documents.sqlite",
        work / "passages",
        work / "vectors" if dense else None,
        work / "models/embedding" if dense else None,
        None if args.no_rerank else work / "models/reranker",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "index", "embed", "search", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--work-dir", type=Path, default=Path("work"))
        if name in {"search", "evaluate"}:
            command.add_argument("--method", choices=("bm25", "dense", "hybrid"), default="hybrid")
            command.add_argument("--no-rerank", action="store_true")
            command.add_argument(
                "--feedback", choices=("none", "question", "counts", "followup"), default="none"
            )
            command.add_argument("--feedback-url")
            command.add_argument("--feedback-model", default="Qwen/Qwen3-32B")
        if name == "search":
            command.add_argument("--question", required=True)
        if name == "evaluate":
            command.add_argument(
                "--split", choices=("development", "heldout", "all"), default="development"
            )
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.work_dir
    if args.command == "download":
        corpus.prepare(work / "data")
        (work / "models").mkdir(parents=True, exist_ok=True)
        index.prepare_tokenizer(work / "models/tokenizer")
        embedding.prepare(work / "models/embedding")
        rerank.prepare(work / "models/reranker")
    elif args.command == "index":
        corpus.build_index(work / "data", work / "documents.sqlite")
        index.prepare(
            work / "data", work / "models/tokenizer", work / "documents.sqlite", work / "passages"
        )
    elif args.command == "embed":
        embedding.encode(
            work / "passages", work / "data", work / "models/embedding", work / "vectors"
        )
    else:
        if args.feedback != "none" and not args.feedback_url:
            parser.error("--feedback requires --feedback-url pointing to a hosted chat model")
        generator = (
            ChatEndpoint(args.feedback_url, args.feedback_model, os.environ.get("FEEDBACK_API_KEY"))
            if args.feedback != "none"
            else None
        )
        retriever = load_retriever(args)
        if args.command == "search":
            result = retriever.search(args.question, args.method, args.feedback, generator)
            result["documents"] = [
                {
                    key: doc[key]
                    for key in ("corpus_row", "doc_id", "title", "score", "reranker_score")
                    if key in doc
                }
                for doc in result["documents"]
            ]
            print(json.dumps(result, indent=2))
        else:
            if args.output.exists():
                raise FileExistsError("Choose a new evaluation output path")
            questions = json.loads((work / "data/questions.json").read_text())
            heldout = set(
                json.loads((Path(__file__).parent / "split.json").read_text())["heldout_ids"]
            )
            if args.split != "all":
                questions = [
                    q
                    for q in questions
                    if (q["question_id"] in heldout) == (args.split == "heldout")
                ]
            result = evaluate(retriever, questions, args.method, args.feedback, generator)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            corpus.write_json(args.output, result)
            print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
