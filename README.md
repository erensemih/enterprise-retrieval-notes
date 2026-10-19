# Enterprise Retrieval Notes

Retrieval experiments on EnterpriseRAG-Bench.

## Usage

```bash
python -m pip install -e .
python -m enterprise_retrieval download --work-dir work
python -m enterprise_retrieval index --work-dir work
python -m enterprise_retrieval search --work-dir work \
  --question "What are the file upload size limits?"
```

Reference-document recall is evaluated separately from answer correctness.

BM25 candidates can be reranked with the pinned BGE cross-encoder.

Documents are partitioned into shared passages to retain full source coverage.
