"""Central configuration for paths and pipeline tunables."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR: Path = PROJECT_ROOT / "chroma"

CORPUS_GLOB: str = "*.md"
REQUIRED_METADATA_FIELDS: tuple[str, ...] = ("doc_name", "section", "version", "status")

# Assignment brief: 3–4 primary source documents of 500–800 words.
# Extra investor PDFs stay indexed for eval queries that are not in that
# set (modern-slavery pass rate, Independent Director terms). Word counts
# are whitespace tokens of Document.body (frontmatter excluded).
PRIMARY_WORD_MIN: int = 500
PRIMARY_WORD_MAX: int = 800
CORPUS_ROLE_PRIMARY: str = "primary"
CORPUS_ROLE_SUPPLEMENTAL: str = "supplemental"
CORPUS_ROLE_FIXTURE: str = "fixture"
ALLOWED_CORPUS_ROLES: frozenset[str] = frozenset(
    {CORPUS_ROLE_PRIMARY, CORPUS_ROLE_SUPPLEMENTAL, CORPUS_ROLE_FIXTURE}
)
PRIMARY_SOURCE_FILES: frozenset[str] = frozenset(
    {
        "posh_policy.md",
        "whistleblower_policy.md",
        "ehs_policy.md",
        "human_rights_policy_v2.md",
    }
)
SUPPLEMENTAL_SOURCE_FILES: frozenset[str] = frozenset(
    {
        "supplier_code_of_conduct.md",
        "nomination_remuneration_policy.md",
        "modern_slavery_statement.md",
        "board_diversity_policy.md",
        "csr_esg_policy.md",
    }
)
FIXTURE_SOURCE_FILE: str = "human_rights_policy_v1.md"

# Static recursive hierarchical splitter. Size is a budget, not a saw.
# Paragraph p90 in this corpus is 411 chars; 500 keeps typical clauses intact.
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100
# Parallel list items are near-duplicates of each other: "Net zero by 2040",
# "Water positive by 2040", and "Zero waste to landfill by 2040" share a
# template, so packing two of them into one window buries the term that tells
# them apart and neither BM25 nor the cross-encoder can pick a winner. A bullet
# at or above this length is already a self-contained clause and is indexed
# alone. Shorter bullets still pack: corpus bullets run 72-284 chars with a
# median of 123 and p75 of 180, and a 90-char fragment is too little context to
# answer from on its own.
LIST_ATOMIC_MIN_CHARS: int = 200
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

# MS MARCO logits. Captured window: Wi-Fi / out-of-domain sits near -11;
# Privilege Leave clauses sit about +2 to +6. 0.0 drops the negative
# tail and keeps those clauses. Compared in safety.py, not in search().
MIN_RERANK_SCORE: float = 0.0
# A query with fewer than this many alphanumeric tokens must contain
# that token in the chunk. "PTO" is one token; supplier "leave" is not.
SHORT_QUERY_MAX_TOKENS: int = 2

# Recall@K uses the window search() returns after rerank, which is
# RETRIEVE_K (5). RERANK_CANDIDATE_K (20) is only the pool that gets
# rescored. A passage the cross-encoder left outside those five never
# reaches the extractive answer, so Recall at 20 would credit hits the
# caller does not see. Answer accuracy is a separate check on the cited
# top passage: a gold chunk at rank 3 counts for recall and still fails
# accuracy. Per-query recall is 0 or 1 because each question has one
# supporting passage; the harness mean is the average of those values.
EVAL_K: int = RETRIEVE_K

# Data-quality fixture: Human Rights v1 stays in the index on purpose.
# Public Coforge investor policies do not publish a PTO handbook; the stale v1
# clause still answers "how many Privilege Leave / PTO days?" until diagnosis.
PLANTED_DOC_NAME: str = "Human Rights Policy"
CURRENT_POLICY_VERSION: str = "2.0"
LEGACY_POLICY_VERSION: str = "1.0"
LEGACY_PTO_DAYS: int = 15
CURRENT_HR_COMPLAINTS_EMAIL: str = "All.HR@coforge.com"
LEGACY_HR_COMPLAINTS_EMAIL: str = "hr.helpdesk@niit-tech.com"

# Live answers use the OpenAI SDK (optional extra ``llm``). The key and model
# tag are never hard-coded; generator_from_env() reads them from the environment.
LLM_API_KEY_ENV: str = "OPENAI_API_KEY"
LLM_MODEL_ENV: str = "RAG_LAB_LLM_MODEL"
# Any server that speaks the OpenAI chat-completions API can answer: hosted
# OpenAI when this is unset, or a local runtime such as Ollama
# (http://localhost:11434/v1), llama.cpp, or vLLM when it is set. Keeping the
# endpoint in the environment means the corpus, retrieval, and eval gold stay
# provider-independent.
LLM_BASE_URL_ENV: str = "RAG_LAB_LLM_BASE_URL"
# Local runtimes do not authenticate, but the SDK refuses to construct a client
# without a key. A base URL therefore stands in for the key. A hosted endpoint
# with no base URL still requires a real key, so this is not a way to skip auth.
LLM_LOCAL_API_KEY: str = "local"
# Temperature 0 keeps the grounded JSON contract stable.
LLM_TEMPERATURE: float = 0.0
LLM_TIMEOUT_SECONDS: float = 120.0
