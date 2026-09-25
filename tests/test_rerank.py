"""Cross-encoder reranking of the fused hybrid candidate list.

Store tests use a scripted embedder, a fake pair scorer, and ``tmp_path`` so
they do not download ``ms-marco-MiniLM-L-6-v2`` or share the repo ``chroma/``
directory. The semantic test is marked ``integration``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from rag_lab import config
from rag_lab.chunking import Chunk, ChunkType, chunk_corpus
from rag_lab.corpus import load_corpus
from rag_lab.exceptions import RetrievalError
from rag_lab.hybrid import HybridHit, HybridRetriever
from rag_lab.index import PolicyIndex
from rag_lab.rerank import (
    CrossEncoderReranker,
    RerankingRetriever,
    rerank_hits,
)


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


class _LengthEmbedder:
    """Stable 3-d vectors so a full-corpus store test does not load MiniLM."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.5]


class _RecordingScorer:
    """Fake pair scorer. Records every batch and never loads a model."""

    def __init__(self, score_for_text: Callable[[str], float]) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self._score_for_text = score_for_text

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, tuple(texts)))
        return [float(self._score_for_text(text)) for text in texts]


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


def _meta(doc_name: str, *, status: str = "current", version: str = "2.0") -> dict[str, str]:
    return {
        "doc_name": doc_name,
        "section": "Fair Wages and Remuneration",
        "version": version,
        "status": status,
    }


def _board_and_legacy_chunks() -> list[Chunk]:
    """Four lexical board hits, one legacy PTO clause, one dense tail."""
    boards = [
        _chunk(
            f"board-{index}",
            f"The board reviews director tenure and sitting fees clause {index}.",
        )
        for index in range(4)
    ]
    legacy = _chunk(
        "legacy-pto",
        "Full-time employees are entitled to 15 days of Privilege Leave (PTO) per calendar year.",
        status="legacy",
        version="1.0",
        doc_name="Human Rights Policy",
    )
    noise = _chunk("annual-report", "Annual report publication calendar for investors.")
    return [*boards, legacy, noise]


def _board_vectors() -> list[list[float]]:
    """Closest vectors are the board clauses. Legacy is inside a depth-5 window."""
    return [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.7, 0.3, 0.0],
        [0.2, 0.8, 0.0],
        [0.0, 1.0, 0.0],
    ]


def test_defaults_rescore_the_fusion_pool_and_return_retrieve_k() -> None:
    """20 fused hits are rescored; 5 are returned. One CPU batch covers the pool."""
    assert config.RERANK_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.RERANK_CANDIDATE_K == 20
    assert config.RETRIEVE_K == 5
    assert config.RERANK_CANDIDATE_K > config.RETRIEVE_K
    assert config.RERANK_BATCH_SIZE >= config.RERANK_CANDIDATE_K


def test_reranker_rejects_blank_model_name_before_loading_weights() -> None:
    with pytest.raises(RetrievalError, match="empty"):
        CrossEncoderReranker(model_name="  ")


def test_cross_encoder_rejects_blank_inputs_before_loading_weights() -> None:
    reranker = CrossEncoderReranker()
    assert reranker.model_name == config.RERANK_MODEL
    with pytest.raises(RetrievalError, match="empty"):
        reranker.score_pairs("   ", ["Privilege Leave"])
    with pytest.raises(RetrievalError, match="empty"):
        reranker.score_pairs("Privilege Leave", [])
    with pytest.raises(RetrievalError, match="empty"):
        reranker.score_pairs("Privilege Leave", ["  "])


def test_equal_scores_keep_fused_order() -> None:
    hits = [
        HybridHit("b", "second clause", 0.02, _meta("B"), 2, 1, 0.2, 3.0),
        HybridHit("a", "first clause", 0.03, _meta("A"), 1, 2, 0.1, 1.0),
    ]

    class _Tie:
        def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
            assert query == "pto"
            assert len(texts) == 2
            return [1.0, 1.0]

    ranked = rerank_hits("pto", hits, _Tie(), limit=2)
    assert [hit.chunk_id for hit in ranked] == ["b", "a"]
    assert ranked[0].score == pytest.approx(1.0)
    assert ranked[0].rrf_score == pytest.approx(0.02)


def test_cross_encoder_score_replaces_rrf_order() -> None:
    """A larger logit wins. It is not added to the tiny RRF value."""
    legacy_meta = _meta("Human Rights Policy", status="legacy", version="1.0")
    hits = [
        HybridHit("high-rrf", "current policy text", 0.05, _meta("Current"), 1, 1, 0.1, 4.0),
        HybridHit("low-rrf", "15 days of Privilege Leave", 0.01, legacy_meta, None, 4, None, 1.0),
    ]

    class _BoostLegacy:
        def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
            assert len(texts) == 2
            return [0.1, 8.5]

    ranked = rerank_hits(
        "How many Privilege Leave / PTO days?",
        hits,
        _BoostLegacy(),
        limit=1,
    )
    assert len(ranked) == 1
    assert ranked[0].chunk_id == "low-rrf"
    assert ranked[0].score == pytest.approx(8.5)
    assert ranked[0].rrf_score == pytest.approx(0.01)
    assert ranked[0].metadata["status"] == "legacy"
    assert ranked[0].metadata["version"] == config.LEGACY_POLICY_VERSION


def test_rerank_hits_rejects_bad_scores() -> None:
    hit = HybridHit("a", "Privilege Leave", 0.02, _meta("A"), 1, 1, 0.1, 1.0)

    class _Short:
        def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
            del query, texts
            return [1.0, 2.0]

    class _Nan:
        def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
            del query, texts
            return [float("nan")]

    class _Bool:
        def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
            del query, texts
            return [True]  # type: ignore[list-item]

    with pytest.raises(RetrievalError, match="scores"):
        rerank_hits("pto", [hit], _Short(), limit=1)
    with pytest.raises(RetrievalError, match="Non-finite"):
        rerank_hits("pto", [hit], _Nan(), limit=1)
    with pytest.raises(RetrievalError, match="Non-numeric"):
        rerank_hits("pto", [hit], _Bool(), limit=1)
    with pytest.raises(RetrievalError, match="Invalid k"):
        rerank_hits("pto", [hit], _Short(), limit=0)
    with pytest.raises(RetrievalError, match="empty"):
        rerank_hits("  ", [hit], _Short(), limit=1)


def test_default_scorer_is_the_ms_marco_cross_encoder(tmp_path: Path) -> None:
    index = PolicyIndex(tmp_path / "chroma", embedder=_ScriptedEmbedder([], [1.0, 0.0, 0.0]))
    retriever = RerankingRetriever(HybridRetriever(index))
    assert isinstance(retriever.scorer, CrossEncoderReranker)
    assert retriever.scorer.model_name == config.RERANK_MODEL


def test_rerank_promotes_legacy_buried_by_fusion(tmp_path: Path) -> None:
    """Legacy sits outside the fused top-2 and inside the scored pool, then wins."""
    chunks = _board_and_legacy_chunks()
    index = PolicyIndex(
        tmp_path / "chroma",
        embedder=_ScriptedEmbedder(_board_vectors(), [1.0, 0.0, 0.0]),
    )
    hybrid = HybridRetriever(index)
    hybrid.upsert(chunks)
    query = "director tenure sitting fees"
    fused_window = hybrid.search(query, k=2)
    fused_pool = hybrid.search(query, k=5)
    assert all(hit.metadata["status"] != "legacy" for hit in fused_window)
    legacy_in_pool = [hit for hit in fused_pool if hit.metadata["status"] == "legacy"]
    assert len(legacy_in_pool) == 1
    assert legacy_in_pool[0].rrf_score < fused_window[0].rrf_score

    scorer = _RecordingScorer(lambda text: 4.0 if f"{config.LEGACY_PTO_DAYS} days" in text else 0.2)
    retriever = RerankingRetriever(hybrid, scorer=scorer, candidate_k=5)
    hits = retriever.search(query, k=2)

    assert len(scorer.calls) == 1
    assert scorer.calls[0][0] == query
    assert len(scorer.calls[0][1]) == len(fused_pool) == 5
    assert any(f"{config.LEGACY_PTO_DAYS} days" in text for text in scorer.calls[0][1])
    assert hits[0].chunk_id == "legacy-pto"
    assert hits[0].score == pytest.approx(4.0)
    assert hits[0].rrf_score == pytest.approx(legacy_in_pool[0].rrf_score)
    assert hits[0].metadata["doc_name"] == config.PLANTED_DOC_NAME
    assert hits[0].metadata["section"] == "Fair Wages and Remuneration"
    assert hits[0].metadata["version"] == config.LEGACY_POLICY_VERSION
    assert hits[0].metadata["status"] == "legacy"
    assert f"{config.LEGACY_PTO_DAYS} days" in hits[0].text
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_legacy_is_scored_before_the_top_k_cut(tmp_path: Path) -> None:
    """A losing legacy row is still passed to the scorer. The cut happens after."""
    chunks = _board_and_legacy_chunks()
    index = PolicyIndex(
        tmp_path / "chroma",
        embedder=_ScriptedEmbedder(_board_vectors(), [1.0, 0.0, 0.0]),
    )
    hybrid = HybridRetriever(index)
    hybrid.upsert(chunks)
    query = "director tenure sitting fees"
    scorer = _RecordingScorer(
        lambda text: -3.0 if f"{config.LEGACY_PTO_DAYS} days" in text else 1.0
    )
    hits = RerankingRetriever(hybrid, scorer=scorer, candidate_k=5).search(query, k=2)
    scored = scorer.calls[0][1]
    assert any(f"{config.LEGACY_PTO_DAYS} days" in text for text in scored)
    assert len(scored) > len(hits)
    assert all(hit.metadata["status"] != "legacy" for hit in hits)
    assert hits[0].score == pytest.approx(1.0)


def test_corpus_pool_is_rescored_and_legacy_pto_can_rank_first(tmp_path: Path) -> None:
    """Full corpus, fake scorer: 20 fused hits in, 5 out, v1 can take rank 1."""
    chunks = chunk_corpus(load_corpus())
    assert len(chunks) > config.RERANK_CANDIDATE_K
    assert any(chunk.metadata["status"] == "legacy" for chunk in chunks)
    index = PolicyIndex(tmp_path / "chroma", embedder=_LengthEmbedder())
    hybrid = HybridRetriever(index)
    hybrid.upsert(chunks)
    query = "How many Privilege Leave / PTO days do I get?"
    scorer = _RecordingScorer(
        lambda text: 10.0 if f"{config.LEGACY_PTO_DAYS} days" in text else 0.0
    )
    retriever = RerankingRetriever(hybrid, scorer=scorer)
    hits = retriever.search(query, k=config.RETRIEVE_K)

    assert len(scorer.calls) == 1
    scored = scorer.calls[0][1]
    assert len(scored) == config.RERANK_CANDIDATE_K
    assert any(f"{config.LEGACY_PTO_DAYS} days" in text for text in scored)
    assert len(hits) == config.RETRIEVE_K
    assert hits[0].score == pytest.approx(10.0)
    assert hits[1].score == pytest.approx(0.0)
    assert hits[0].metadata["doc_name"] == config.PLANTED_DOC_NAME
    assert hits[0].metadata["version"] == config.LEGACY_POLICY_VERSION
    assert hits[0].metadata["status"] == "legacy"
    assert hits[0].metadata["section"]
    assert "Privilege Leave (PTO)" in hits[0].text
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_rerank_validates_query_and_empty_index(tmp_path: Path) -> None:
    index = PolicyIndex(tmp_path / "chroma", embedder=_ScriptedEmbedder([], [1.0, 0.0, 0.0]))
    scorer = _RecordingScorer(lambda text: 1.0)
    retriever = RerankingRetriever(HybridRetriever(index), scorer=scorer)
    with pytest.raises(RetrievalError, match="empty"):
        retriever.search("  ")
    with pytest.raises(RetrievalError, match="Invalid k"):
        retriever.search("Privilege Leave", k=0)
    with pytest.raises(RetrievalError, match="tokens"):
        retriever.search("???")
    assert retriever.search("Privilege Leave") == []
    assert scorer.calls == []
    with pytest.raises(RetrievalError, match="candidate_k"):
        RerankingRetriever(HybridRetriever(index), candidate_k=0)


@pytest.mark.integration
def test_cross_encoder_rerank_surfaces_legacy_pto(tmp_path: Path) -> None:
    """Load → chunk → hybrid fusion → MS MARCO MiniLM still returns Human Rights v1."""
    chunks = chunk_corpus(load_corpus())
    assert any(chunk.metadata["status"] == "legacy" for chunk in chunks)
    index = PolicyIndex(tmp_path / "chroma")
    hybrid = HybridRetriever(index)
    hybrid.upsert(chunks)
    retriever = RerankingRetriever(hybrid)
    assert isinstance(retriever.scorer, CrossEncoderReranker)
    assert retriever.scorer.model_name == config.RERANK_MODEL
    hits = retriever.search(
        "How many Privilege Leave / PTO days do I get?",
        k=config.RETRIEVE_K,
    )
    assert hits
    assert len(hits) == config.RETRIEVE_K
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)
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
    rendered = [
        f"{hit.metadata['doc_name']} v{hit.metadata['version']} "
        f"{hit.metadata['status']} score={hit.score:.3f} section={hit.metadata['section']}"
        for hit in hits
    ]
    assert legacy_hits, (
        "PTO query must return Human Rights Policy v1.0 inside reranked top-k; got "
        + "; ".join(rendered)
    )
    assert any("Privilege Leave (PTO)" in hit.text for hit in legacy_hits)
    index.close()
