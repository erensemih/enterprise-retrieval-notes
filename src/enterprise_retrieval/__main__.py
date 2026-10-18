"""Index and search the enterprise document corpus."""

import argparse
import json
from pathlib import Path

from . import corpus
from .evaluation import evaluate
from .search import Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["download", "index", "search", "evaluate"]:
        command = commands.add_parser(name)
        command.add_argument("--work-dir", type=Path, default=Path("work"))
        if name in {"search", "evaluate"}:
            command.add_argument("--method", choices=["bm25"], default="bm25")
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
    elif args.command == "index":
        corpus.build_index(work / "data", work / "documents.sqlite")
    else:
        generator = None
        retriever = Retriever(work / "documents.sqlite")
        if args.command == "search":
            result = retriever.search(args.question, args.method, "none", generator)
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
            result = evaluate(retriever, questions, args.method, "none", generator)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            corpus.write_json(args.output, result)


if __name__ == "__main__":
    main()
