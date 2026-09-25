# RAG Lab

A production-grade Retrieval-Augmented Generation pipeline built from scratch.

## Roadmap

| Part | Module | Status |
|------|--------|--------|
| 1 | Coforge investor policies + planted Human Rights v1 PTO conflict | ✅ |
| 2 | Chunking + sentence-transformers + ChromaDB | ⬜ |
| 3 | Hybrid search (dense + BM25) with Reciprocal Rank Fusion | ⬜ |
| 4 | Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) | ⬜ |
| 5 | PyTest harness: 8+ queries, Recall@K, answer accuracy | ⬜ |
| 6 | Diagnose planted defect (2-Question Debugging Framework) | ⬜ |
| 7 | Source attribution (doc, section, version) | ⬜ |
| 8 | GitHub Actions CI | ⬜ |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Layout

```
docs/architecture.md   Part 2 design lock (chunk size, overlap, pipeline)
src/rag_lab/           pipeline package (one module per part)
tests/                 pytest suite; test_minimal_loop.py gates Parts 3+
data/raw/              Coforge policies (versioned; planted Human Rights v1)
chroma/                vector store (gitignored, rebuildable)
```

Design freeze before Part 2 code: [docs/architecture.md](docs/architecture.md).
