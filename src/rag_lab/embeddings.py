"""Sentence embeddings for dense policy retrieval.

``all-MiniLM-L6-v2`` is a 6-layer, 384-dimensional MiniLM. It was trained with
cosine similarity, so this module L2-normalizes every vector. On unit vectors,
cosine similarity equals the inner product. Chroma then stores cosine distance
as ``1 - cosine`` (lower is closer). Model id, dimension, device, and batch size
come from ``rag_lab.config``.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from rag_lab.config import (
    EMBED_BATCH_SIZE,
    EMBED_DEVICE,
    EMBED_DIM,
    EMBED_MODEL,
    EMBED_NORMALIZE,
)
from rag_lab.exceptions import IndexingError

logger = logging.getLogger(__name__)


class TextEmbedder(Protocol):
    """Encodes documents and queries into one vector space."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per document text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Return the vector for a single query."""
        ...


class MiniLMEmbedder:
    """Lazy loader for ``sentence-transformers/all-MiniLM-L6-v2``."""

    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        """Bind a Hugging Face model id. Weights load on the first encode.

        Raises:
            IndexingError: If ``model_name`` is blank.
        """
        if not model_name.strip():
            raise IndexingError("Embedding model name is empty")
        self.model_name = model_name
        self._model: Any | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed policy chunks.

        Raises:
            IndexingError: If ``texts`` is empty, a row is blank, or encoding fails.
        """
        if not texts:
            raise IndexingError("Cannot embed an empty text list")
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise IndexingError(f"Cannot embed empty text at index {index}")
        model = self._load()
        try:
            encoded: Any = model.encode(
                texts,
                batch_size=EMBED_BATCH_SIZE,
                normalize_embeddings=EMBED_NORMALIZE,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise IndexingError(
                f"Failed to embed {len(texts)} texts with {self.model_name}"
            ) from exc
        return _coerce_matrix(encoded, expected_rows=len(texts), model_name=self.model_name)

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string.

        Raises:
            IndexingError: If ``text`` is empty or encoding fails.
        """
        if not text.strip():
            raise IndexingError("Cannot embed an empty query")
        return self.embed_documents([text])[0]

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise IndexingError(
                f"sentence-transformers is not installed; cannot load {self.model_name}"
            ) from exc
        logger.info("Loading embedding model %s on %s", self.model_name, EMBED_DEVICE)
        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=EMBED_DEVICE,
                similarity_fn_name="cosine",
            )
        except Exception as exc:
            raise IndexingError(f"Failed to load embedding model {self.model_name!r}") from exc
        return self._model


def _coerce_matrix(encoded: Any, *, expected_rows: int, model_name: str) -> list[list[float]]:
    raw: Any = encoded.tolist() if hasattr(encoded, "tolist") else encoded
    if not isinstance(raw, list):
        raise IndexingError(f"{model_name} returned a non-list embedding payload")
    if len(raw) != expected_rows:
        raise IndexingError(
            f"{model_name} returned {len(raw)} vectors for {expected_rows} texts"
        )
    matrix: list[list[float]] = []
    for row_index, row in enumerate(raw):
        if not isinstance(row, list):
            raise IndexingError(f"Vector {row_index} from {model_name} is not a list")
        if len(row) != EMBED_DIM:
            raise IndexingError(
                f"Expected embedding dim {EMBED_DIM}, got {len(row)} from {model_name}"
            )
        matrix.append([float(value) for value in row])
    return matrix
