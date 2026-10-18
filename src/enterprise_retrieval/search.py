from pathlib import Path

from .corpus import search as corpus_search
from .rerank import Reranker


class Retriever:
    def __init__(self, corpus_index, reranker_model=None):
        self.corpus_index = Path(corpus_index)
        if not self.corpus_index.is_file():
            raise FileNotFoundError("Build the document index first")
        self.reranker = None
        if reranker_model is not None:
            self.reranker = Reranker(Path(reranker_model))

    def search(self, question, method="bm25", feedback="none", generate=None, top_k=5):
        if method != "bm25" or feedback != "none":
            raise ValueError("Only BM25 retrieval is available")
        candidates = corpus_search(self.corpus_index, question, 50)
        if self.reranker is not None:
            candidates, _ = self.reranker.rank(question, candidates, "candidate_rank")
        return {"documents": candidates[:top_k], "feedback": {}}
