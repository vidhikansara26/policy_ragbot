"""Cross-encoder reranking of the hybrid candidate list.

``cross-encoder/ms-marco-MiniLM-L-6-v2`` scores each (query, chunk) pair with
one relevance value. Hybrid fusion only knows ranks: a chunk's RRF score is
``1 / (RRF_K + rank)`` summed across the dense and BM25 lists. The
cross-encoder reads the query and the chunk together, so a clause that fusion
placed below a near-miss can move up when the pair actually answers the
question.

Depth is two numbers. ``RERANK_CANDIDATE_K`` (20) fused hits are rescored.
That is the pool ``HybridRetriever`` already built from lists of depth
``HYBRID_CANDIDATE_K``. ``RETRIEVE_K`` (5) hits are returned, the same answer
window dense and hybrid search use, which is the K the eval harness will
measure. The extra scores exist only to order that window.

Twenty is the fusion cutoff. A chunk outside it already lost to twenty others
on reciprocal rank. Scoring those tail pairs would run the transformer over
passages MiniLM and BM25 both ranked below the pool. Scoring only the five
hits that would have been returned cannot promote a fused rank of 6 into the
answer window, which is the reason to rerank.

The cross-encoder score replaces the order. It is not added to ``rrf_score``.
RRF values sit near 0.03; an MS MARCO logit is often several units wide.
Adding them would make the logit the whole decision and leave RRF as rounding
error. Batch min-max is the same trap fusion avoids: the worst pair in the
pool would become 0 whenever the pool changes. Sort the model scores.

``status=legacy`` is not a filter. Every fused hit is scored, then the list
is cut to ``k``. Human Rights Policy v1 stays eligible for the stale
Privilege Leave / PTO query.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rag_lab.chunking import Chunk
from rag_lab.config import (
    RERANK_BATCH_SIZE,
    RERANK_CANDIDATE_K,
    RERANK_DEVICE,
    RERANK_MODEL,
    RETRIEVE_K,
)
from rag_lab.exceptions import RetrievalError
from rag_lab.hybrid import HybridHit, HybridRetriever

logger = logging.getLogger(__name__)


class PairScorer(Protocol):
    """Scores query–chunk pairs. Higher means more relevant."""

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return one score per text, aligned with ``texts``.

        Raises:
            RetrievalError: If the query, the texts, or the model call is invalid.
        """
        ...


@dataclass(frozen=True)
class RerankedHit:
    """One reranked hit. ``score`` is the cross-encoder value, higher is better.

    ``rrf_score`` is the fused score from before the reorder. Ranks are the
    original dense and BM25 positions. ``score`` does not include ``rrf_score``.
    """

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, str]
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None
    dense_distance: float | None
    bm25_score: float | None


class CrossEncoderReranker:
    """Lazy loader for ``cross-encoder/ms-marco-MiniLM-L-6-v2``."""

    def __init__(self, model_name: str = RERANK_MODEL) -> None:
        """Bind a Hugging Face model id. Weights load on the first ``score_pairs``.

        Raises:
            RetrievalError: If ``model_name`` is blank or the batch size is invalid.
        """
        if not model_name.strip():
            raise RetrievalError("Rerank model name is empty")
        if RERANK_BATCH_SIZE <= 0:
            raise RetrievalError(f"Invalid rerank batch size {RERANK_BATCH_SIZE}")
        if not RERANK_DEVICE.strip():
            raise RetrievalError("Rerank device is empty")
        self.model_name = model_name
        self._model: Any | None = None

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        """Score ``(query, text)`` pairs. Higher is more relevant.

        This checkpoint's activation is identity, so the values are logits.
        They are sorted as returned and are not mixed with the RRF value.

        Raises:
            RetrievalError: If ``query`` is empty, a chunk is blank, the model
                cannot be loaded, or the score payload is the wrong shape.
        """
        _require_query_and_texts(query, texts)
        model = self._load()
        pairs = [(query, text) for text in texts]
        try:
            encoded: Any = model.predict(
                pairs,
                batch_size=RERANK_BATCH_SIZE,
                show_progress_bar=False,
                apply_softmax=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise RetrievalError(
                f"Failed to score {len(texts)} pairs with {self.model_name}"
            ) from exc
        return _coerce_scores(encoded, expected=len(texts), model_name=self.model_name)

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RetrievalError(
                f"sentence-transformers is not installed; cannot load {self.model_name}"
            ) from exc
        logger.info("Loading rerank model %s on %s", self.model_name, RERANK_DEVICE)
        try:
            self._model = CrossEncoder(self.model_name, device=RERANK_DEVICE)
        except Exception as exc:
            raise RetrievalError(f"Failed to load rerank model {self.model_name!r}") from exc
        return self._model


def rerank_hits(
    query: str,
    hits: Sequence[HybridHit],
    scorer: PairScorer,
    *,
    limit: int,
) -> list[RerankedHit]:
    """Score every fused hit, then return the top ``limit`` by that score.

    Equal scores keep fused order. No metadata filter is applied: a
    ``status=legacy`` row is scored with the rest of the pool and can occupy
    a returned slot when its pair score is high enough.

    Raises:
        RetrievalError: If ``query`` is empty, ``limit`` is not positive, a
            chunk body is blank, or ``scorer`` returns a bad score vector.
    """
    if limit <= 0:
        raise RetrievalError(f"Invalid k={limit}")
    if not hits:
        if not query.strip():
            raise RetrievalError("Query text is empty")
        return []
    _require_query_and_texts(query, [hit.text for hit in hits])
    scores = _scores_for(query, hits, scorer)
    order = sorted(range(len(hits)), key=lambda index: (-scores[index], index))
    ranked: list[RerankedHit] = []
    for index in order[:limit]:
        hit = hits[index]
        ranked.append(
            RerankedHit(
                chunk_id=hit.chunk_id,
                text=hit.text,
                score=scores[index],
                metadata=dict(hit.metadata),
                rrf_score=hit.rrf_score,
                dense_rank=hit.dense_rank,
                bm25_rank=hit.bm25_rank,
                dense_distance=hit.dense_distance,
                bm25_score=hit.bm25_score,
            )
        )
    return ranked


class RerankingRetriever:
    """Hybrid search followed by a cross-encoder reorder of the fused pool.

    ``search`` asks the hybrid retriever for ``max(k, candidate_k)`` fused
    hits, scores every one of them, and returns ``k``. Legacy rows stay in
    the pool through scoring.
    """

    def __init__(
        self,
        hybrid: HybridRetriever,
        *,
        scorer: PairScorer | None = None,
        candidate_k: int = RERANK_CANDIDATE_K,
    ) -> None:
        """Bind a hybrid retriever and the pair scorer.

        ``scorer=None`` uses :class:`CrossEncoderReranker`. Weights stay unloaded
        until the first non-empty search.

        Raises:
            RetrievalError: If ``candidate_k`` is not positive.
        """
        if candidate_k <= 0:
            raise RetrievalError(f"Invalid candidate_k={candidate_k}")
        self._hybrid = hybrid
        self._scorer: PairScorer = scorer if scorer is not None else CrossEncoderReranker()
        self._candidate_k = candidate_k

    @property
    def hybrid(self) -> HybridRetriever:
        """Fused dense + BM25 retriever, before the cross-encoder reorder."""
        return self._hybrid

    @property
    def scorer(self) -> PairScorer:
        """Pair scorer used to reorder fused hits."""
        return self._scorer

    def upsert(self, chunks: list[Chunk]) -> None:
        """Store ``chunks`` on the hybrid retriever. Does not drop ``status=legacy``.

        Raises:
            IndexingError: If the dense upsert or the BM25 rebuild fails.
            RetrievalError: If the stored collection cannot be read back.
        """
        self._hybrid.upsert(chunks)

    def search(self, query: str, *, k: int = RETRIEVE_K) -> list[RerankedHit]:
        """Return the top ``k`` chunks after cross-encoder reranking.

        The hybrid stage is asked for ``max(k, candidate_k)`` fused hits.
        With the defaults that is 20 rescored and 5 returned. Every fused hit
        is scored, including ``status=legacy``, and the score sort is cut to
        ``k``.

        Raises:
            RetrievalError: If ``query`` is empty, has no tokens, ``k`` is
                invalid, fusion fails, or the pair scorer fails.
        """
        if k <= 0:
            raise RetrievalError(f"Invalid k={k}")
        depth = max(k, self._candidate_k)
        fused = self._hybrid.search(query, k=depth)
        if not fused:
            return []
        ranked = rerank_hits(query, fused, self._scorer, limit=k)
        logger.info(
            "Reranked %s of %s fused hits (candidate_k=%s)",
            len(ranked),
            len(fused),
            self._candidate_k,
        )
        return ranked


def _require_query_and_texts(query: str, texts: Sequence[str]) -> None:
    if not query.strip():
        raise RetrievalError("Query text is empty")
    if not texts:
        raise RetrievalError("Cannot rerank an empty candidate list")
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise RetrievalError(f"Cannot rerank empty chunk text at index {index}")


def _scores_for(query: str, hits: Sequence[HybridHit], scorer: PairScorer) -> list[float]:
    texts = [hit.text for hit in hits]
    try:
        raw = scorer.score_pairs(query, texts)
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(f"Reranker failed on {len(texts)} candidates") from exc
    if len(raw) != len(hits):
        raise RetrievalError(f"Reranker returned {len(raw)} scores for {len(hits)} candidates")
    return [_as_score(value, index=index) for index, value in enumerate(raw)]


def _as_score(value: object, *, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RetrievalError(f"Non-numeric rerank score at index {index}")
    score = float(value)
    if not math.isfinite(score):
        raise RetrievalError(f"Non-finite rerank score at index {index}")
    return score


def _coerce_scores(encoded: Any, *, expected: int, model_name: str) -> list[float]:
    raw: Any = encoded.tolist() if hasattr(encoded, "tolist") else encoded
    if not isinstance(raw, list):
        raise RetrievalError(f"{model_name} returned a non-list score payload")
    if len(raw) != expected:
        raise RetrievalError(f"{model_name} returned {len(raw)} scores for {expected} pairs")
    scores: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, list):
            if len(value) != 1:
                raise RetrievalError(f"{model_name} returned {len(value)} labels at index {index}")
            value = value[0]
        scores.append(_as_score(value, index=index))
    return scores
