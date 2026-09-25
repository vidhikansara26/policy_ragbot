"""Retrieve-loop gate: load, chunk, MiniLM embed, Chroma store, dense query.

The semantic test downloads ``all-MiniLM-L6-v2`` and is marked ``integration``.
Store-contract tests use a deterministic fake embedder and a ``tmp_path`` Chroma
directory so they do not download weights or share the repo ``chroma/`` dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab import config
from rag_lab.chunking import Chunk, ChunkType, chunk_corpus
from rag_lab.corpus import load_corpus
from rag_lab.embeddings import MiniLMEmbedder
from rag_lab.exceptions import IndexingError, RetrievalError
from rag_lab.index import PolicyIndex


class _FakeEmbedder:
    """Stable 3-d vectors so Chroma tests do not load MiniLM."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.5]


def test_embedder_rejects_blank_model_name() -> None:
    with pytest.raises(IndexingError, match="empty"):
        MiniLMEmbedder(model_name="  ")


def test_embedder_rejects_blank_text_before_loading_weights() -> None:
    embedder = MiniLMEmbedder()
    with pytest.raises(IndexingError, match="empty"):
        embedder.embed_documents([])
    with pytest.raises(IndexingError, match="empty"):
        embedder.embed_query("   ")


def test_collection_is_persistent_cosine(tmp_path: Path) -> None:
    index = PolicyIndex(tmp_path / "chroma", embedder=_FakeEmbedder())
    assert index.collection_name == config.CHROMA_COLLECTION
    assert index.distance == "cosine"
    assert index.collection_metadata["hnsw:space"] == "cosine"
    assert index.persist_directory == tmp_path / "chroma"
    assert index.persist_directory.is_dir()


def test_upsert_keeps_legacy_human_rights_metadata(tmp_path: Path) -> None:
    chunks = chunk_corpus(load_corpus())
    legacy_pto = [
        chunk
        for chunk in chunks
        if chunk.metadata["status"] == "legacy" and "Privilege Leave (PTO)" in chunk.text
    ]
    assert legacy_pto, "stale Human Rights v1 PTO clause must be chunked"
    index = PolicyIndex(tmp_path / "chroma", embedder=_FakeEmbedder())
    index.upsert(chunks)
    assert index.count() == len(chunks)
    stored = index.get_chunk(legacy_pto[0].chunk_id)
    assert stored.metadata["doc_name"] == config.PLANTED_DOC_NAME
    assert stored.metadata["section"]
    assert stored.metadata["version"] == config.LEGACY_POLICY_VERSION
    assert stored.metadata["status"] == "legacy"
    assert f"{config.LEGACY_PTO_DAYS} days" in stored.text


def test_upsert_rejects_empty_batch(tmp_path: Path) -> None:
    index = PolicyIndex(tmp_path / "chroma", embedder=_FakeEmbedder())
    with pytest.raises(IndexingError, match="empty"):
        index.upsert([])


def test_search_validates_query_and_k(tmp_path: Path) -> None:
    index = PolicyIndex(tmp_path / "chroma", embedder=_FakeEmbedder())
    with pytest.raises(RetrievalError, match="empty"):
        index.search("  ")
    with pytest.raises(RetrievalError, match="Invalid k"):
        index.search("Privilege Leave", k=0)
    assert index.search("Privilege Leave") == []


def test_reopen_persistent_client_keeps_legacy_row(tmp_path: Path) -> None:
    chunks = chunk_corpus(load_corpus())
    legacy = next(chunk for chunk in chunks if chunk.metadata["status"] == "legacy")
    directory = tmp_path / "chroma"
    first = PolicyIndex(directory, embedder=_FakeEmbedder())
    first.upsert([legacy])
    first.close()
    assert (directory / "chroma.sqlite3").is_file()
    second = PolicyIndex(directory, embedder=_FakeEmbedder())
    assert second.count() == 1
    stored = second.get_chunk(legacy.chunk_id)
    assert stored.metadata["status"] == "legacy"
    assert stored.metadata["version"] == config.LEGACY_POLICY_VERSION


def test_upsert_rejects_missing_attribution(tmp_path: Path) -> None:
    index = PolicyIndex(tmp_path / "chroma", embedder=_FakeEmbedder())
    chunk = Chunk(
        chunk_id="missing-status",
        text="Privilege Leave text",
        chunk_type=ChunkType.PARAGRAPH,
        metadata={
            "doc_name": "Human Rights Policy",
            "section": "Fair Wages",
            "version": "1.0",
        },
    )
    with pytest.raises(IndexingError, match="status"):
        index.upsert([chunk])


@pytest.mark.integration
def test_minimal_retrieve_loop_surfaces_legacy_pto(tmp_path: Path) -> None:
    """Load → chunk → MiniLM embed → Chroma → dense query still returns v1 PTO."""
    documents = load_corpus()
    chunks = chunk_corpus(documents)
    assert any(chunk.metadata["status"] == "legacy" for chunk in chunks)

    index = PolicyIndex(tmp_path / "chroma")
    assert isinstance(index.embedder, MiniLMEmbedder)
    assert index.embedder.model_name == config.EMBED_MODEL
    index.upsert(chunks)
    assert index.count() == len(chunks)

    hits = index.search(
        "How many Privilege Leave / PTO days do I get?",
        k=config.RETRIEVE_K,
    )
    assert hits
    assert len(hits) <= config.RETRIEVE_K
    assert [hit.distance for hit in hits] == sorted(hit.distance for hit in hits)
    for hit in hits:
        assert hit.metadata["doc_name"]
        assert hit.metadata["section"]
        assert hit.metadata["version"]
        assert hit.metadata["status"]

    legacy_hits = [
        hit
        for hit in hits
        if hit.metadata["doc_name"] == config.PLANTED_DOC_NAME
        and hit.metadata["version"] == config.LEGACY_POLICY_VERSION
        and hit.metadata["status"] == "legacy"
    ]
    assert legacy_hits, "PTO query must retrieve Human Rights Policy v1.0 inside top-k"
    assert any("Privilege Leave (PTO)" in hit.text for hit in legacy_hits)
