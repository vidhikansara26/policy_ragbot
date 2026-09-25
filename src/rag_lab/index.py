"""Persistent Chroma index for dense policy retrieval.

The collection ``coforge_policies`` uses cosine distance. MiniLM vectors are
L2-normalized, so cosine similarity equals the inner product and Chroma's
cosine distance is ``1 - cosine`` (lower is closer).

Chunks are stored with attribution metadata. ``status=legacy`` is not filtered:
Human Rights Policy v1 is a data-quality fixture and must remain retrievable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_lab.chunking import Chunk
from rag_lab.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHROMA_DISTANCE,
    REQUIRED_METADATA_FIELDS,
    RETRIEVE_K,
)
from rag_lab.embeddings import MiniLMEmbedder, TextEmbedder
from rag_lab.exceptions import IndexingError, RetrievalError

logger = logging.getLogger(__name__)

_ALLOWED_DISTANCES = frozenset({"cosine", "l2", "ip"})


@dataclass(frozen=True)
class SearchHit:
    """One dense-retrieval hit. ``distance`` is cosine distance (lower is closer)."""

    chunk_id: str
    text: str
    distance: float
    metadata: dict[str, str]


@dataclass(frozen=True)
class StoredChunk:
    """A chunk read back from the index, including attribution metadata."""

    chunk_id: str
    text: str
    metadata: dict[str, str]


class _PrecomputedEmbeddings:
    """Reject Chroma's built-in embedder so only MiniLM vectors are stored.

    Chroma still records an embedding-function name on the collection. This
    adapter satisfies that contract and refuses to encode, so a second model
    cannot silently replace ``all-MiniLM-L6-v2``.
    """

    @staticmethod
    def name() -> str:
        return "precomputed-minilm"

    def __call__(self, input: list[str]) -> list[list[float]]:
        raise IndexingError(
            "Chroma's embedding function is disabled; "
            f"refusing to embed {len(input)} texts. Pass MiniLM vectors explicitly"
        )

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> _PrecomputedEmbeddings:
        if config:
            raise IndexingError(
                f"precomputed-minilm embedding function does not take config: {config}"
            )
        return _PrecomputedEmbeddings()

    def is_legacy(self) -> bool:
        return False


class PolicyIndex:
    """Cosine Chroma collection persisted under ``CHROMA_DIR`` (or a test path)."""

    def __init__(
        self,
        persist_directory: Path | None = None,
        *,
        embedder: TextEmbedder | None = None,
        collection_name: str = CHROMA_COLLECTION,
        distance: str = CHROMA_DISTANCE,
    ) -> None:
        """Open or create the persistent collection.

        Raises:
            IndexingError: If the path, collection name, or distance is invalid,
                or Chroma cannot be opened.
        """
        if not collection_name.strip():
            raise IndexingError("Chroma collection name is empty")
        if distance not in _ALLOWED_DISTANCES:
            raise IndexingError(f"Unsupported Chroma distance {distance!r}")
        directory = persist_directory or CHROMA_DIR
        if directory.exists() and not directory.is_dir():
            raise IndexingError(f"Chroma path is not a directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        self.persist_directory = directory
        self.collection_name = collection_name
        self.distance = distance
        self.embedder: TextEmbedder = embedder if embedder is not None else MiniLMEmbedder()
        self._client, self._collection = _open_collection(
            directory,
            collection_name=collection_name,
            distance=distance,
        )

    @property
    def collection_metadata(self) -> dict[str, str]:
        """Return collection metadata, including ``hnsw:space``."""
        raw: Any = self._collection.metadata or {}
        if not isinstance(raw, dict):
            raise IndexingError(f"Collection {self.collection_name} metadata is not a dict")
        return {str(key): str(value) for key, value in raw.items()}

    def upsert(self, chunks: list[Chunk]) -> None:
        """Embed ``chunks`` and store them. Does not drop ``status=legacy``.

        Raises:
            IndexingError: If ``chunks`` is empty, metadata is incomplete, or
                the store rejects the batch.
        """
        if not chunks:
            raise IndexingError("Cannot index an empty chunk list")
        ids = [chunk.chunk_id for chunk in chunks]
        if any(not chunk_id.strip() for chunk_id in ids):
            raise IndexingError("Chunk is missing chunk_id")
        if len(ids) != len(set(ids)):
            raise IndexingError("Duplicate chunk_id in upsert batch")
        documents = [chunk.text for chunk in chunks]
        if any(not text.strip() for text in documents):
            raise IndexingError("Refusing to index an empty chunk body")
        # Legacy Human Rights v1 stays indexed; filtering it hides the PTO incident.
        metadatas = [_metadata_for_chroma(chunk.metadata) for chunk in chunks]
        embeddings = self.embedder.embed_documents(documents)
        if len(embeddings) != len(chunks):
            raise IndexingError(
                f"Embedder returned {len(embeddings)} vectors for {len(chunks)} chunks"
            )
        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise IndexingError(
                f"Failed to upsert {len(chunks)} chunks into {self.collection_name}"
            ) from exc
        logger.info(
            "Upserted %s chunks into collection %s at %s",
            len(chunks),
            self.collection_name,
            self.persist_directory,
        )

    def count(self) -> int:
        """Return how many chunks are stored.

        Raises:
            IndexingError: If Chroma cannot count the collection.
        """
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise IndexingError(f"Failed to count collection {self.collection_name}") from exc

    def get_chunk(self, chunk_id: str) -> StoredChunk:
        """Fetch one stored chunk by id.

        Raises:
            RetrievalError: If ``chunk_id`` is blank or not in the collection.
        """
        if not chunk_id.strip():
            raise RetrievalError("chunk_id is empty")
        try:
            result: Any = self._collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise RetrievalError(f"Failed to fetch chunk {chunk_id}") from exc
        payload = _as_mapping(result, failure=f"Failed to fetch chunk {chunk_id}")
        ids: list[Any] = list(payload.get("ids") or [])
        documents: list[Any] = list(payload.get("documents") or [])
        metadatas: list[Any] = list(payload.get("metadatas") or [])
        if not ids:
            raise RetrievalError(f"Chunk not found: {chunk_id}")
        document = documents[0]
        metadata = metadatas[0]
        if not isinstance(document, str) or not isinstance(metadata, dict):
            raise RetrievalError(f"Chroma returned an incomplete record for {chunk_id}")
        return StoredChunk(
            chunk_id=str(ids[0]),
            text=document,
            metadata=_stringify_metadata(metadata),
        )

    def search(self, query: str, *, k: int = RETRIEVE_K) -> list[SearchHit]:
        """Return the ``k`` nearest chunks by cosine distance.

        No metadata filter is applied. A Privilege Leave query can therefore
        still surface Human Rights Policy v1 (``status=legacy``).

        Raises:
            RetrievalError: If ``query`` is empty, ``k`` is invalid, or Chroma fails.
        """
        if not query.strip():
            raise RetrievalError("Query text is empty")
        if k <= 0:
            raise RetrievalError(f"Invalid k={k}")
        try:
            total = self.count()
        except IndexingError as exc:
            raise RetrievalError(f"Failed to query collection {self.collection_name}") from exc
        if total == 0:
            return []
        vector = self.embedder.embed_query(query)
        try:
            result: Any = self._collection.query(
                query_embeddings=[vector],
                n_results=min(k, total),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise RetrievalError(f"Failed to query collection {self.collection_name}") from exc
        hits = _hits_from_query(result)
        logger.info("Retrieved %s hits from collection %s", len(hits), self.collection_name)
        return hits

    def close(self) -> None:
        """Release the persistent client so the directory can be reopened.

        Raises:
            IndexingError: If the client cannot be closed.
        """
        try:
            self._client.close()
        except Exception as exc:
            raise IndexingError(
                f"Failed to close Chroma client at {self.persist_directory}"
            ) from exc


def _open_collection(
    persist_directory: Path,
    *,
    collection_name: str,
    distance: str,
) -> tuple[Any, Any]:
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:
        raise IndexingError("chromadb is not installed") from exc
    try:
        client: Any = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        collection: Any = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance},
            embedding_function=_PrecomputedEmbeddings(),
        )
    except IndexingError:
        raise
    except Exception as exc:
        raise IndexingError(
            f"Failed to open Chroma collection {collection_name} at {persist_directory}"
        ) from exc
    metadata: Any = collection.metadata or {}
    space = metadata.get("hnsw:space") if isinstance(metadata, dict) else None
    if space != distance:
        raise IndexingError(
            f"Collection {collection_name} uses distance {space!r}, expected {distance!r}"
        )
    return client, collection


def _metadata_for_chroma(metadata: dict[str, str]) -> dict[str, str]:
    missing = [
        field for field in REQUIRED_METADATA_FIELDS if not metadata.get(field, "").strip()
    ]
    if missing:
        raise IndexingError(f"Chunk metadata missing required fields: {missing}")
    cleaned: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(value, str) or not value.strip():
            raise IndexingError(f"Metadata {key!r} must be a non-empty string")
        cleaned[key] = value
    return cleaned


def _stringify_metadata(metadata: dict[Any, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in metadata.items()}


def _as_mapping(result: Any, *, failure: str) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    raise RetrievalError(failure)


def _hits_from_query(result: Any) -> list[SearchHit]:
    payload = _as_mapping(result, failure="Chroma query returned an unexpected payload")
    id_rows: list[Any] = list(payload.get("ids") or [])
    doc_rows: list[Any] = list(payload.get("documents") or [])
    meta_rows: list[Any] = list(payload.get("metadatas") or [])
    dist_rows: list[Any] = list(payload.get("distances") or [])
    if not id_rows or not id_rows[0]:
        return []
    ids = list(id_rows[0])
    documents = list(doc_rows[0]) if doc_rows else []
    metadatas = list(meta_rows[0]) if meta_rows else []
    distances = list(dist_rows[0]) if dist_rows else []
    if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
        raise RetrievalError("Chroma query returned ragged hit columns")
    hits: list[SearchHit] = []
    for chunk_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances, strict=True
    ):
        if not isinstance(document, str) or not isinstance(metadata, dict) or distance is None:
            raise RetrievalError(f"Chroma returned an incomplete hit for {chunk_id}")
        hits.append(
            SearchHit(
                chunk_id=str(chunk_id),
                text=document,
                distance=float(distance),
                metadata=_stringify_metadata(metadata),
            )
        )
    return hits
