"""Central configuration for paths and pipeline tunables."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR: Path = PROJECT_ROOT / "chroma"

CORPUS_GLOB: str = "*.md"
REQUIRED_METADATA_FIELDS: tuple[str, ...] = ("doc_name", "section", "version", "status")

# Static recursive hierarchical splitter. Size is a budget, not a saw.
# Paragraph p90 in this corpus is 391 chars; 500 keeps typical clauses intact.
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100
# all-MiniLM-L6-v2 is 384-d and trained for cosine similarity.
# L2-normalized vectors make cosine identical to the inner product.
EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM: int = 384
EMBED_BATCH_SIZE: int = 32
EMBED_NORMALIZE: bool = True
EMBED_DEVICE: str = "cpu"
CHROMA_COLLECTION: str = "coforge_policies"
CHROMA_DISTANCE: str = "cosine"
RETRIEVE_K: int = 5

# Okapi BM25 (Robertson / Sparck Jones). k1 saturates repeated terms; b pulls
# long chunks toward the corpus average length. 1.5 and 0.75 are the usual
# defaults. There is no stopword list: "not" and "days" separate the stale
# 15-day Privilege Leave clause from the current "does not set a numeric
# entitlement" wording.
BM25_K1: float = 1.5
BM25_B: float = 0.75

# Minimum depth of each ranked list before fusion. Wider than RETRIEVE_K so a
# lexical hit just outside the dense top-5 still enters RRF. A search that
# asks for more than this fetches max(k, HYBRID_CANDIDATE_K) from each list.
HYBRID_CANDIDATE_K: int = 20

# Cormack, Clarke, and Buettcher, SIGIR 2009. Reciprocal Rank Fusion uses
# 1 / (RRF_K + rank) and ignores raw scores. Cosine distance sits in [0, 2]
# while BM25 is an unbounded idf sum, so adding the two numbers lets the BM25
# magnitude swamp a stronger dense rank. Min-max rescaling a short candidate
# list is just as brittle: the worst hit becomes 0 and the best becomes 1
# whenever the candidate set changes. k=60 keeps rank 1 and rank 2 close.
RRF_K: int = 60

# MS MARCO passage ranker. A 6-layer MiniLM reads the query and the chunk in
# one forward pass and emits a single relevance score. It is a different
# checkpoint from the bi-encoder: all-MiniLM-L6-v2 never sees the pair together.
RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_DEVICE: str = "cpu"
# 32 covers the default fused pool in one CPU batch.
RERANK_BATCH_SIZE: int = 32

# How many fused hits the cross-encoder rescores. This matches the hybrid
# candidate pool (HYBRID_CANDIDATE_K): fusion already kept these 20, and the
# reranker exists to reorder that pool. Rescoring only RETRIEVE_K would shuffle
# the five hits we were going to return and could not promote a chunk RRF
# placed just outside that window. Rescoring the whole collection repeats work
# on chunks both retrievers already left below the fusion cutoff. search()
# still returns RETRIEVE_K hits; the other scores only order that window.
# A caller who passes a larger k rescores max(k, RERANK_CANDIDATE_K) and
# receives k.
RERANK_CANDIDATE_K: int = 20

# Data-quality fixture: Human Rights v1 stays in the index on purpose.
# Public Coforge investor policies do not publish a PTO handbook; the stale v1
# clause still answers "how many Privilege Leave / PTO days?" until diagnosis.
PLANTED_DOC_NAME: str = "Human Rights Policy"
CURRENT_POLICY_VERSION: str = "2.0"
LEGACY_POLICY_VERSION: str = "1.0"
LEGACY_PTO_DAYS: int = 15
CURRENT_HR_COMPLAINTS_EMAIL: str = "All.HR@coforge.com"
LEGACY_HR_COMPLAINTS_EMAIL: str = "hr.helpdesk@niit-tech.com"
