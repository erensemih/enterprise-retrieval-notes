# Enterprise Retrieval Notes

Retrieval over 511,962 company documents from
[EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench),
with two experiments on using the first search to improve the next:

- **Keyword expansion:** word-match counts from BM25 results were given to an LLM
  to suggest additional search terms, with a question-only control.
- **Missing-information search:** five retrieved excerpts were read by an LLM,
  which could request one focused follow-up search.

The baseline combines BM25, Qwen3 embeddings and a BGE reranker. Both feedback
approaches are implemented in [feedback.py](src/enterprise_retrieval/feedback.py).

## Results

The development comparison used 401 questions, of which 375 had scorable references.
Recall@5 is the fraction of reference documents retrieved in the first five results,
averaged across scorable questions.

| Method | Reference recall@5 |
| --- | ---: |
| BM25 | 62.22% |
| BGE-base dense | 42.32% |
| Qwen3-Embedding-0.6B dense | 46.74% |
| BM25 + Qwen | 65.21% |
| BM25 + BGE reranker | 68.05% |
| BM25 + Qwen + BGE reranker | **71.82%** |

The selected hybrid reached **70.94%** on 94 scorable questions from a separate
99-question test. The combined result was **71.65%** on 469 scorable questions
out of 500, including development data. The test set has now been used.

Short keyword expansion helped BM25 with reranking, but added no clear gain to
the hybrid. The follow-up search also produced no clear aggregate gain.
[Experiment results](docs/experiments.md) include the controls and unsuccessful
attempts. These are retrieval scores; answer correctness was not measured.

## Usage

Python 3.10+, SQLite FTS5 and CUDA are required for the complete pipeline.

```bash
python -m pip install -e .
python -m enterprise_retrieval download --work-dir work
python -m enterprise_retrieval index --work-dir work
python -m enterprise_retrieval embed --work-dir work
python -m enterprise_retrieval search --work-dir work \
  --question "What are the file upload size limits?"
```

Index construction and embedding belong on a compute worker. Models and data
are downloaded separately. Each document is split into nonoverlapping passages
of 510 payload tokens using the pinned BGE tokenizer. Dense search uses the same
character spans, exact cosine similarity and the best passage per document.
The top 50 BM25 and dense candidates are fused with equal RRF60 weights, then
reranked to return five documents.

For CPU-only search after passage indexing, use `--method bm25 --no-rerank`.
To try a feedback approach, add `--feedback counts` or `--feedback followup`
(the latter uses `--method bm25`) and `--feedback-url` pointing to a hosted
chat-completions endpoint. The experiments used Qwen3-32B without thinking;
the endpoint must support `chat_template_kwargs` and the model's JSON responses.
Use `--feedback question` for the question-only expansion control.

```bash
python -m enterprise_retrieval evaluate --work-dir work \
  --split development --output work/evaluation.json
python -m pip install -e '.[dev]'
python -m unittest discover -s tests
```

## Performance measurements

On 100 sequential queries after five warm-ups, retrieval took **5.84 s median**
and **8.91 s p95** on one A100. Models and vectors were resident; SQLite indices
were staged on RAM-backed tmpfs. BM25 took 5.10 s median and reranking 0.70 s.
Setup, network and answer generation are excluded. The passage BM25 index is
4.84 GB; the vector matrix is 6.56 GB.

The [source package](src/enterprise_retrieval) contains the pipeline and prompts;
[tests](tests) cover ranking, source identity, label isolation and evaluation.
Code is [MIT licensed](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md).
