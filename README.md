# Policy RAG Platform

Enterprise Retrieval-Augmented Generation over versioned Coforge policy documents.
Built as incremental production capabilities. The dense retrieve loop gates
hybrid search; hybrid search gates cross-encoder reranking; reranking gates
the evaluation harness; the harness gates incident diagnosis.

## Delivery status

| Phase | Capability | Status |
|-------|------------|--------|
| 1 | Versioned corpus + stale Human Rights v1 (PTO conflict) | Done |
| 2 | Hierarchical chunking + MiniLM + Chroma + minimal retrieve | Done |
| 3 | Hybrid search (dense + BM25) with Reciprocal Rank Fusion | Done |
| 4 | Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) | Done |
| 5 | Eval harness: 8+ queries, Recall@K, answer accuracy | Done |
| 6 | Data-quality diagnosis (2-Question Debugging Framework) | Next |
| 7 | Source attribution (doc, section, version) | Metadata on chunks |
| 8 | GitHub Actions CI | Pending |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Check from the command line

One question runs the whole path (chunk, MiniLM, Chroma, BM25, reciprocal rank fusion, cross-encoder) and prints the returned chunks plus the cited answer:

```bash
python -m rag_lab query "How many Privilege Leave / PTO days do I get?"
```

The eval scoreboard prints Recall@K and answer accuracy separately. The Privilege Leave row records Human Rights Policy v1 as the data-quality fixture when that passage is retrieved:

```bash
python -m rag_lab eval
```

The first run loads the embedding and rerank models and writes `chroma/`. Offline tests do not.

## Layout

```
docs/architecture.md   locked ingest/chunk/index design
src/rag_lab/           platform package (one module per capability)
tests/                 pytest; test_minimal_loop.py gates hybrid search
data/raw/              Coforge policies (includes stale Human Rights v1)
chroma/                vector store (gitignored, rebuildable)
```

Design reference: [docs/architecture.md](docs/architecture.md).
