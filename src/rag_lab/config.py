"""Central configuration for paths and pipeline tunables."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR: Path = PROJECT_ROOT / "chroma"

CORPUS_GLOB: str = "*.md"
REQUIRED_METADATA_FIELDS: tuple[str, ...] = ("doc_name", "section", "version", "status")

# Static recursive hierarchical splitter (Part 2). Size is a budget, not a saw.
# Paragraph p90 in this corpus is 391 chars; 500 keeps typical clauses intact.
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100
EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION: str = "coforge_policies"
RETRIEVE_K: int = 5

# Planted data-quality defect (Part 1 / Part 6): Human Rights v1 stays in the index.
# Official Coforge investor policies do not publish a PTO handbook; the conflict is
# a retired v1 clause that still answers "how many Privilege Leave / PTO days?"
PLANTED_DOC_NAME: str = "Human Rights Policy"
CURRENT_POLICY_VERSION: str = "2.0"
LEGACY_POLICY_VERSION: str = "1.0"
LEGACY_PTO_DAYS: int = 15
CURRENT_HR_COMPLAINTS_EMAIL: str = "All.HR@coforge.com"
LEGACY_HR_COMPLAINTS_EMAIL: str = "hr.helpdesk@niit-tech.com"
