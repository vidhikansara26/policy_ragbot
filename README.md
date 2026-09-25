# Policy RAG Platform

Enterprise Retrieval-Augmented Generation over versioned Coforge policy documents.
Built as incremental production capabilities with a hard retrieve-loop gate before
hybrid search or reranking.

## Delivery status

| Phase | Capability | Status |
|-------|------------|--------|
| 1 | Versioned corpus + stale Human Rights v1 (PTO conflict) | Done |
| 2 | Hierarchical chunking + MiniLM + Chroma + minimal retrieve | Done |
| 3 | Hybrid search (dense + BM25) with Reciprocal Rank Fusion | Blocked on retrieve loop |
| 4 | Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) | Pending |
| 5 | Eval harness: 8+ queries, Recall@K, answer accuracy | Pending |
| 6 | Data-quality diagnosis (2-Question Debugging Framework) | Pending |
| 7 | Source attribution (doc, section, version) | Metadata on chunks |
| 8 | GitHub Actions CI | Pending |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Layout

```
docs/architecture.md   locked ingest/chunk/index design
src/rag_lab/           platform package (one module per capability)
tests/                 pytest; test_minimal_loop.py gates hybrid search
data/raw/              Coforge policies (includes stale Human Rights v1)
chroma/                vector store (gitignored, rebuildable)
```

Design reference: [docs/architecture.md](docs/architecture.md).
