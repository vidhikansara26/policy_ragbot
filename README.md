# Policy RAG Platform

Enterprise Retrieval-Augmented Generation over versioned Coforge policy documents.
Built as incremental production capabilities. The dense retrieve loop gates
hybrid search; hybrid search gates cross-encoder reranking; reranking gates
the evaluation harness; the harness gates incident diagnosis. Grounded answer
generation (`generate.py` / `answer` CLI) is available after rerank and does
not change extractive `query` or eval scoring.

## Delivery status

| Phase | Capability | Status |
|-------|------------|--------|
| 1 | Versioned corpus + stale Human Rights v1 (PTO conflict) | Done |
| 2 | Hierarchical chunking + MiniLM + Chroma + minimal retrieve | Done |
| 3 | Hybrid search (dense + BM25) with Reciprocal Rank Fusion | Done |
| 4 | Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) | Done |
| 5 | Eval harness: 8+ queries, Recall@K, answer accuracy | Done |
| 6 | Data-quality diagnosis (2-Question Debugging Framework) | Next |
| 7 | Grounded generation + source attribution (doc, section, version) | Done |
| 8 | GitHub Actions CI | Pending |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set OPENAI_API_KEY and RAG_LAB_LLM_MODEL for the answer command
pytest
```

## Check from the command line

One question runs the whole path (chunk, MiniLM, Chroma, BM25, reciprocal rank fusion, cross-encoder) and prints the returned chunks plus the extractive cited answer:

```bash
python -m rag_lab query "How many Privilege Leave / PTO days do I get?"
```

Generate a grounded answer (requires `OPENAI_API_KEY` and `RAG_LAB_LLM_MODEL`,
for example `gpt-4o-mini`). Install the SDK with `pip install -e ".[dev,llm]"`.
The model sees only the reranked window. Citations keep the chunk section
string. A current/legacy disagreement is answered from the current policy and
the legacy line is marked. The planted 15-day entitlement is not published as
current policy. `status=legacy` is still not filtered at ingest. `query` stays
extractive:

```bash
export OPENAI_API_KEY=sk-...
export RAG_LAB_LLM_MODEL=gpt-4o-mini
python -m rag_lab answer "How many Privilege Leave / PTO days do I get?"
```

The eval scoreboard prints `recall_at_k` and `extractive_answer_accuracy`, then `generated_key_fact`, `citation_complete`, `groundedness`, and `conflict_handling`. The Privilege Leave retrieval row still records Human Rights Policy v1 as the data-quality fixture. Extractive accuracy on that row means the rank-1 chunk is v1. Key-fact accuracy on the same question means the published answer used the current wording and cited v2 while flagging v1. `eval` calls the configured LLM:

```bash
python -m rag_lab eval
```

Print dense-only and reciprocal-rank-fusion hits on the same rows. The cross-encoder is not called. `All.HR@coforge.com` is the MiniLM miss in this corpus: dense leaves Human Rights Policy v2, section 13. Grievance Redressal, outside the top 5 (dense rank 8), BM25 ranks that chunk 1, and RRF places it in the fused top 5.

```bash
python -m rag_lab compare "All.HR@coforge.com"
```

The first run loads the embedding and rerank models and writes `chroma/`. Offline tests do not.

## Layout

```
docs/architecture.md   locked ingest/chunk/index design
docs/corpus.md         source PDFs, roles, 500–800 word band
src/rag_lab/           platform package (one module per capability)
tests/                 pytest; test_minimal_loop.py gates hybrid search
data/raw/              Coforge policies (includes stale Human Rights v1)
chroma/                vector store (gitignored, rebuildable)
```

Design reference: [docs/architecture.md](docs/architecture.md).
Corpus inventory: [docs/corpus.md](docs/corpus.md).
