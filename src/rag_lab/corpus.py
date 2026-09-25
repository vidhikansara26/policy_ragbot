"""Load versioned markdown policy documents from ``data/raw``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_lab.config import CORPUS_GLOB, RAW_DATA_DIR, REQUIRED_METADATA_FIELDS
from rag_lab.exceptions import DataLoadError


@dataclass(frozen=True)
class Document:
    """A single policy document with source metadata for later attribution."""

    path: Path
    body: str
    metadata: dict[str, str]

    @property
    def doc_name(self) -> str:
        return self.metadata["doc_name"]

    @property
    def section(self) -> str:
        return self.metadata["section"]

    @property
    def version(self) -> str:
        return self.metadata["version"]

    @property
    def status(self) -> str:
        return self.metadata["status"]


def _parse_frontmatter(raw: str, *, path: Path) -> tuple[dict[str, str], str]:
    """Split YAML-like ``key: value`` frontmatter from the markdown body.

    Raises:
        DataLoadError: If the file is missing frontmatter, required fields, or a body.
    """
    if not raw.startswith("---"):
        raise DataLoadError(f"{path} is missing YAML frontmatter")

    rest = raw[3:].lstrip("\n")
    end = rest.find("\n---")
    if end == -1:
        raise DataLoadError(f"{path} has an unclosed frontmatter block")

    header = rest[:end]
    body = rest[end + 4 :].strip()
    if not body:
        raise DataLoadError(f"{path} has an empty body")

    metadata: dict[str, str] = {}
    for line_no, line in enumerate(header.splitlines(), start=2):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise DataLoadError(f"{path}:{line_no} is not a key: value pair")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            raise DataLoadError(f"{path}:{line_no} has an empty key or value")
        metadata[key] = value

    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing:
        raise DataLoadError(f"{path} is missing required metadata: {missing}")
    return metadata, body


def load_corpus(directory: Path | None = None) -> list[Document]:
    """Load every markdown policy under ``directory`` (defaults to ``RAW_DATA_DIR``).

    Raises:
        DataLoadError: If the directory is missing, empty, or a file is malformed.
    """
    root = directory or RAW_DATA_DIR
    if not root.is_dir():
        raise DataLoadError(f"Corpus directory does not exist: {root}")

    paths = sorted(root.glob(CORPUS_GLOB))
    if not paths:
        raise DataLoadError(f"No markdown documents found in {root}")

    documents: list[Document] = []
    for path in paths:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DataLoadError(f"Failed to read {path}") from exc
        metadata, body = _parse_frontmatter(raw, path=path)
        documents.append(Document(path=path, body=body, metadata=metadata))
    return documents
