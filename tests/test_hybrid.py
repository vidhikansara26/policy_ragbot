"""Hybrid search: Okapi BM25 plus dense hits, fused with Reciprocal Rank Fusion.

Store tests use a scripted embedder and ``tmp_path`` so they do not download
MiniLM or share the repo ``chroma/`` directory. The semantic test is marked
``integration``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab import config
from rag_lab.bm25 import BM25Index, SparseHit
from rag_lab.chunking import Chunk, ChunkType, chunk_corpus
from rag_lab.corpus import load_corpus
from rag_lab.exceptions import IndexingError, RetrievalError
from rag_lab.hybrid import HybridRetriever, fuse_hits
from rag_lab.index import PolicyIndex, SearchHit


class _ScriptedEmbedder:
    """Document vectors in upsert order, plus one fixed query vector."""

    def __init__(
        self,
        document_vectors: list[list[float]],
        query_vector: list[float],
    ) -> None:
        self._document_vectors = document_vectors
        self._query_vector = query_vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) != len(self._document_vectors):
            raise AssertionError(f"expected {len(self._document_vectors)} texts, got {len(texts)}")
        return [list(vector) for vector in self._document_vectors]

    def embed_query(self, text: str) -> list[float]:
        del text
        return list(self._query_vector)


def _chunk(
    chunk_id: str,
    text: str,
    *,
    status: str = "current",
    version: str = "2.0",
    doc_name: str = "Board Diversity Policy",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        chunk_type=ChunkType.PARAGRAPH,
        metadata={
            "doc_name": doc_name,
            "section": "Fair Wages and Remuneration",
            "version": version,
            "status": status,
        },
    )


def _meta(doc_name: str) -> dict[str, str]:
    return {
        "doc_name": doc_name,
        "section": "Fair Wages and Remuneration",
        "version": "2.0",
        "status": "current",
    }


def test_rrf_prefers_agreement_over_raw_score_addition() -> None:
    """A huge BM25 number must not beat a document both lists rank near the top.

    Adding ``(1 - distance) + bm25`` would score B at 0.10 + 30 = 30.1 and A at
    0.95 + 1 = 1.95, so B would win on magnitude. RRF uses ranks only:

    - A is dense rank 1 and BM25 rank 2 → 1/61 + 1/62
    - B is dense rank 4 and BM25 rank 1 → 1/64 + 1/61
    """
    dense = [
        SearchHit("a", "both lists", 0.05, _meta("A")),
        SearchHit("pad-1", "dense only", 0.20, _meta("Pad")),
        SearchHit("pad-2", "dense only", 0.40, _meta("Pad")),
        SearchHit("b", "lexical spike", 0.90, _meta("B")),
    ]
    sparse = [
        SparseHit("b", "lexical spike", 30.0, _meta("B")),
        SparseHit("a", "both lists", 1.0, _meta("A")),
    ]
    fused = fuse_hits(dense, sparse, rrf_k=config.RRF_K, limit=4)
    assert [hit.chunk_id for hit in fused] == ["a", "b", "pad-1", "pad-2"]
    assert fused[0].rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1].rrf_score == pytest.approx(1 / 64 + 1 / 61)
    assert fused[0].rrf_score > fused[1].rrf_score
    assert fused[0].dense_rank == 1
    assert fused[0].bm25_rank == 2
    assert fused[1].bm25_score == pytest.approx(30.0)


def test_rrf_tie_breaks_on_chunk_id() -> None:
    dense = [SearchHit("b", "dense", 0.1, _meta("B"))]
    sparse = [SparseHit("a", "sparse", 4.0, _meta("A"))]
    fused = fuse_hits(dense, sparse, rrf_k=60, limit=2)
    assert [hit.chunk_id for hit in fused] == ["a", "b"]
    assert fused[0].dense_rank is None
    assert fused[0].bm25_rank == 1
    assert fused[1].bm25_rank is None
    assert fused[1].dense_distance == pytest.approx(0.1)


def test_fuse_rejects_bad_parameters() -> None:
    with pytest.raises(RetrievalError, match="rrf_k"):
        fuse_hits([], [], rrf_k=-1, limit=5)
    with pytest.raises(RetrievalError, match="Invalid k"):
        fuse_hits([], [], rrf_k=60, limit=0)


def test_bm25_saturates_repeated_terms() -> None:
    index = BM25Index(
        [
            _chunk("once", "pto entitlement"),
            _chunk("repeat", "pto pto pto entitlement"),
        ]
    )
    hits = index.search("pto", k=2)
    assert [hit.chunk_id for hit in hits] == ["repeat", "once"]
    assert hits[0].score > hits[1].score
    assert hits[0].score < hits[1].score * 3


def test_bm25_rejects_invalid_inputs() -> None:
    chunk = _chunk("one", "Privilege Leave")
    with pytest.raises(IndexingError, match="empty"):
        BM25Index([])
    with pytest.raises(IndexingError, match="k1"):
        BM25Index([chunk], k1=0)
    with pytest.raises(IndexingError, match="b="):
        BM25Index([chunk], b=1.5)
    with pytest.raises(IndexingError, match="Duplicate"):
        BM25Index([chunk, _chunk("one", "other words here")])
    with pytest.raises(RetrievalError, match="empty"):
        BM25Index([chunk]).search("  ", k=1)
    with pytest.raises(RetrievalError, match="tokens"):
        BM25Index([chunk]).search("???", k=1)
    with pytest.raises(RetrievalError, match="Invalid k"):
        BM25Index([chunk]).search("pto", k=0)


def test_bm25_ranks_legacy_pto_clause_above_current_denials() -> None:
    """The stale 15-day clause is a lexical match and stays in the sparse index."""
    chunks = chunk_corpus(load_corpus())
    index = BM25Index(chunks)
    hits = index.search("How many Privilege Leave / PTO days do I get?", k=config.RETRIEVE_K)
    assert hits
    top = hits[0]
    assert top.metadata["doc_name"] == config.PLANTED_DOC_NAME
    assert top.metadata["version"] == config.LEGACY_POLICY_VERSION
    assert top.metadata["status"] == "legacy"
    assert f"{config.LEGACY_PTO_DAYS} days" in top.text


def test_hybrid_fuses_lexical_legacy_with_dense_neighbor(tmp_path: Path) -> None:
    """BM25 rank 1 (legacy PTO) and dense rank 1 (no shared terms) both survive."""
    legacy = _chunk(
        "legacy-pto",
        "Full-time employees are entitled to 15 days of Privilege Leave (PTO) per calendar year.",
        status="legacy",
        version="1.0",
        doc_name="Human Rights Policy",
    )
    dense_only = _chunk(
        "dense-board",
        "The board reviews director tenure and sitting fees each year.",
    )
    neither = _chunk(
        "annual-report",
        "Annual report publication calendar for the board.",
    )
    chunks = [legacy, dense_only, neither]
    query = [1.0, 0.0, 0.0]
    vectors = [
        [0.0, 1.0, 0.0],  # legacy: farthest from the query
        [1.0, 0.0, 0.0],  # dense-only: exact cosine match
        [0.6, 0.8, 0.0],  # middle
    ]
    index = PolicyIndex(tmp_path / "chroma", embedder=_ScriptedEmbedder(vectors, query))
    retriever = HybridRetriever(index)
    retriever.upsert(chunks)

    hits = retriever.search("Privilege Leave PTO days", k=3)
    assert [hit.chunk_id for hit in hits] == ["legacy-pto", "dense-board", "annual-report"]
    assert hits[0].metadata["status"] == "legacy"
    assert hits[0].metadata["version"] == config.LEGACY_POLICY_VERSION
    assert hits[0].bm25_rank == 1
    assert hits[0].dense_rank == 3
    assert hits[1].dense_rank == 1
    assert hits[1].bm25_rank is None
    assert f"{config.LEGACY_PTO_DAYS} days" in hits[0].text
    assert [hit.rrf_score for hit in hits] == sorted(
        (hit.rrf_score for hit in hits),
        reverse=True,
    )


def test_bm25_hit_outside_dense_window_still_fuses(tmp_path: Path) -> None:
    """A lexical legacy chunk past the dense candidate cutoff still enters RRF."""
    legacy = _chunk(
        "lexical-pto",
        "Employees receive 15 days of Privilege Leave (PTO).",
        status="legacy",
        version="1.0",
        doc_name="Human Rights Policy",
    )
    distractors = [
        _chunk(f"dense-{index}", f"Board topic number {index} about director tenure.")
        for index in range(5)
    ]
    chunks = distractors + [legacy]
    vectors = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.7, 0.3, 0.0],
        [0.6, 0.4, 0.0],
        [0.0, 1.0, 0.0],
    ]
    index = PolicyIndex(
        tmp_path / "chroma",
        embedder=_ScriptedEmbedder(vectors, [1.0, 0.0, 0.0]),
    )
    retriever = HybridRetriever(index, candidate_k=5)
    retriever.upsert(chunks)
    hits = retriever.search("Privilege Leave PTO", k=2)
    ids = {hit.chunk_id for hit in hits}
    assert "lexical-pto" in ids
    assert "dense-0" in ids
    lexical = next(hit for hit in hits if hit.chunk_id == "lexical-pto")
    assert lexical.metadata["status"] == "legacy"
    assert lexical.dense_rank is None
    assert lexical.bm25_rank == 1


def test_hybrid_rebuilds_bm25_from_persisted_chunks(tmp_path: Path) -> None:
    legacy = _chunk(
        "legacy-pto",
        "Full-time employees are entitled to 15 days of Privilege Leave (PTO).",
        status="legacy",
        version="1.0",
        doc_name="Human Rights Policy",
    )
    other = _chunk("dense-board", "The board reviews director tenure and sitting fees.")
    directory = tmp_path / "chroma"
    vectors = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    query = [1.0, 0.0, 0.0]
    first = PolicyIndex(directory, embedder=_ScriptedEmbedder(vectors, query))
    HybridRetriever(first).upsert([legacy, other])
    first.close()

    reopened = PolicyIndex(directory, embedder=_ScriptedEmbedder([], query))
    hits = HybridRetriever(reopened).search("Privilege Leave PTO days", k=2)
    assert hits[0].chunk_id == "legacy-pto"
    assert hits[0].metadata["status"] == "legacy"
    assert hits[0].metadata["doc_name"] == config.PLANTED_DOC_NAME
    assert hits[0].bm25_rank == 1


def test_hybrid_validates_query_and_empty_index(tmp_path: Path) -> None:
    index = PolicyIndex(tmp_path / "chroma", embedder=_ScriptedEmbedder([], [1.0, 0.0, 0.0]))
    retriever = HybridRetriever(index)
    with pytest.raises(RetrievalError, match="empty"):
        retriever.search("  ")
    with pytest.raises(RetrievalError, match="Invalid k"):
        retriever.search("Privilege Leave", k=0)
    with pytest.raises(RetrievalError, match="tokens"):
        retriever.search("???")
    assert retriever.search("Privilege Leave") == []
    with pytest.raises(RetrievalError, match="candidate_k"):
        HybridRetriever(index, candidate_k=0)
    with pytest.raises(IndexingError, match="BM25"):
        HybridRetriever(index, bm25_k1=0)


@pytest.mark.integration
def test_hybrid_minilm_loop_surfaces_legacy_pto(tmp_path: Path) -> None:
    """Load → chunk → MiniLM + BM25 → RRF still returns Human Rights Policy v1."""
    chunks = chunk_corpus(load_corpus())
    assert any(chunk.metadata["status"] == "legacy" for chunk in chunks)
    index = PolicyIndex(tmp_path / "chroma")
    retriever = HybridRetriever(index)
    retriever.upsert(chunks)
    hits = retriever.search(
        "How many Privilege Leave / PTO days do I get?",
        k=config.RETRIEVE_K,
    )
    assert hits
    assert len(hits) <= config.RETRIEVE_K
    assert [hit.rrf_score for hit in hits] == sorted(
        (hit.rrf_score for hit in hits),
        reverse=True,
    )
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
