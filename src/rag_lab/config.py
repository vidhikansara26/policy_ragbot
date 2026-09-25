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

# Data-quality fixture: Human Rights v1 stays in the index on purpose.
# Public Coforge investor policies do not publish a PTO handbook; the stale v1
# clause still answers "how many Privilege Leave / PTO days?" until diagnosis.
PLANTED_DOC_NAME: str = "Human Rights Policy"
CURRENT_POLICY_VERSION: str = "2.0"
LEGACY_POLICY_VERSION: str = "1.0"
LEGACY_PTO_DAYS: int = 15
CURRENT_HR_COMPLAINTS_EMAIL: str = "All.HR@coforge.com"
LEGACY_HR_COMPLAINTS_EMAIL: str = "hr.helpdesk@niit-tech.com"
