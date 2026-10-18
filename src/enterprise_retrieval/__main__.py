"""Index and search the enterprise document corpus."""

import argparse
import json
from pathlib import Path

from . import corpus
from .search import Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["download", "index", "search"]:
        command = commands.add_parser(name)
        command.add_argument("--work-dir", type=Path, default=Path("work"))
        if name in {"search", "evaluate"}:
            command.add_argument("--method", choices=["bm25"], default="bm25")
        if name == "search":
            command.add_argument("--question", required=True)
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


if __name__ == "__main__":
    main()
