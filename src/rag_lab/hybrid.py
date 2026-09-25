"""Hybrid policy search: MiniLM dense hits plus BM25, fused with RRF.

Dense cosine distance and Okapi BM25 are left as two ranked lists.
``reciprocal_rank_fusion`` scores a chunk as the sum of ``1 / (RRF_K + rank)``
across those lists (Cormack, Clarke, Buettcher, SIGIR 2009). Rank is 1-based
and best-first. A chunk missing from a list adds nothing for that list.

Raw scores stay unused. Cosine distance is bounded (about ``[0, 2]`` for
L2-normalized MiniLM vectors) while BM25 grows with idf and has no shared
ceiling. Adding them lets a large lexical score outrank a document both
retrievers placed near the top. Min-max scaling each list before adding has
the same problem on a short candidate set: the bottom hit is forced to 0 and
the top hit to 1, and that scaling changes whenever the candidate set changes.
RRF depends only on order, and ``RRF_K`` (default 60) keeps rank 1 and rank 2
close so a single first place cannot drown out agreement.

``status=legacy`` is not filtered on either list. Human Rights Policy v1
remains eligible for the stale Privilege Leave / PTO query.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

from rag_lab.bm25 import BM25Index, SparseHit, tokenize
from rag_lab.chunking import Chunk
from rag_lab.config import BM25_B, BM25_K1, HYBRID_CANDIDATE_K, RETRIEVE_K, RRF_K
from rag_lab.exceptions import IndexingError, RetrievalError
from rag_lab.index import PolicyIndex, SearchHit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridHit:
    """One fused hit. ``rrf_score`` is higher-is-better and is not a distance.

    ``dense_rank`` and ``bm25_rank`` are 1-based positions in the candidate
    lists. Either is ``None`` when that retriever did not return the chunk.
    """

    chunk_id: str
    text: str
    rrf_score: float
    metadata: dict[str, str]
    dense_rank: int | None
    bm25_rank: int | None
    dense_distance: float | None
    bm25_score: float | None


def fuse_hits(
    dense_hits: Sequence[SearchHit],
    sparse_hits: Sequence[SparseHit],
    *,
    rrf_k: int,
    limit: int,
) -> list[HybridHit]:
    """Fuse two best-first lists with Reciprocal Rank Fusion.

    List position is the rank. Duplicate ids in one list keep the first
    (best) position. Equal fused scores break on ``chunk_id``.

    Raises:
        RetrievalError: If ``rrf_k`` is negative or ``limit`` is not positive.
    """
    if rrf_k < 0:
        raise RetrievalError(f"Invalid rrf_k={rrf_k}")
    if limit <= 0:
        raise RetrievalError(f"Invalid k={limit}")

    scores: dict[str, float] = {}
    dense_rank: dict[str, int] = {}
    sparse_rank: dict[str, int] = {}
    dense_by_id: dict[str, SearchHit] = {}
    sparse_by_id: dict[str, SparseHit] = {}

    for rank, dense_hit in enumerate(dense_hits, start=1):
        if dense_hit.chunk_id in dense_rank:
            continue
        dense_rank[dense_hit.chunk_id] = rank
        dense_by_id[dense_hit.chunk_id] = dense_hit
        scores[dense_hit.chunk_id] = scores.get(dense_hit.chunk_id, 0.0) + (1.0 / (rrf_k + rank))

    for rank, sparse_hit in enumerate(sparse_hits, start=1):
        if sparse_hit.chunk_id in sparse_rank:
            continue
        sparse_rank[sparse_hit.chunk_id] = rank
        sparse_by_id[sparse_hit.chunk_id] = sparse_hit
        scores[sparse_hit.chunk_id] = scores.get(sparse_hit.chunk_id, 0.0) + (1.0 / (rrf_k + rank))

    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    fused: list[HybridHit] = []
    for chunk_id in ordered[:limit]:
        dense = dense_by_id.get(chunk_id)
        sparse = sparse_by_id.get(chunk_id)
        text, metadata = _hit_payload(dense, sparse, chunk_id=chunk_id)
        fused.append(
            HybridHit(
                chunk_id=chunk_id,
                text=text,
                rrf_score=scores[chunk_id],
                metadata=metadata,
                dense_rank=dense_rank.get(chunk_id),
                bm25_rank=sparse_rank.get(chunk_id),
                dense_distance=None if dense is None else dense.distance,
                bm25_score=None if sparse is None else sparse.score,
            )
        )
    return fused


class HybridRetriever:
    """Dense Chroma search plus in-memory BM25, fused with RRF.

    BM25 is rebuilt from the chunks stored in ``index``. The policy corpus is
    small, so a second on-disk postings file would only risk drifting from
    the vectors. Writes go through :meth:`upsert`, which refreshes both sides
    and does not filter ``status=legacy``.
    """

    def __init__(
        self,
        index: PolicyIndex,
        *,
        rrf_k: int = RRF_K,
        candidate_k: int = HYBRID_CANDIDATE_K,
        bm25_k1: float = BM25_K1,
        bm25_b: float = BM25_B,
    ) -> None:
        """Bind a dense index and the fusion tunables.

        Raises:
            RetrievalError: If ``rrf_k`` is negative or ``candidate_k`` is not positive.
            IndexingError: If the BM25 parameters are outside the Okapi range.
        """
        if rrf_k < 0:
            raise RetrievalError(f"Invalid rrf_k={rrf_k}")
        if candidate_k <= 0:
            raise RetrievalError(f"Invalid candidate_k={candidate_k}")
        if not _valid_bm25_params(k1=bm25_k1, b=bm25_b):
            raise IndexingError(f"Invalid BM25 parameters k1={bm25_k1}, b={bm25_b}")
        self._index = index
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b
        self._bm25: BM25Index | None = None

    def upsert(self, chunks: list[Chunk]) -> None:
        """Store ``chunks`` in Chroma and rebuild BM25 from the full collection.

        Legacy rows stay indexed. The sparse index is the whole collection,
        not only this batch, so a second upsert cannot drop earlier chunks
        from BM25.

        Raises:
            IndexingError: If the dense upsert or the BM25 rebuild fails.
            RetrievalError: If the stored collection cannot be read back.
        """
        self._index.upsert(chunks)
        self._bm25 = self._build_bm25()

    def search(self, query: str, *, k: int = RETRIEVE_K) -> list[HybridHit]:
        """Return the top ``k`` chunks after dense + BM25 Reciprocal Rank Fusion.

        Each retriever contributes ``max(k, candidate_k)`` hits. No metadata
        filter is applied.

        Raises:
            RetrievalError: If ``query`` is empty, has no tokens, ``k`` is
                invalid, or either retriever fails.
        """
        if not query.strip():
            raise RetrievalError("Query text is empty")
        if k <= 0:
            raise RetrievalError(f"Invalid k={k}")
        if not tokenize(query):
            raise RetrievalError("Query has no searchable tokens")
        try:
            total = self._index.count()
        except IndexingError as exc:
            raise RetrievalError(
                f"Failed to query collection {self._index.collection_name}"
            ) from exc
        if total == 0:
            return []
        depth = max(k, self._candidate_k)
        dense_hits = self._index.search(query, k=depth)
        sparse_hits = self._ensure_bm25().search(query, k=depth)
        fused = fuse_hits(dense_hits, sparse_hits, rrf_k=self._rrf_k, limit=k)
        logger.info(
            "Fused %s hybrid hits (dense_candidates=%s, bm25_candidates=%s, rrf_k=%s)",
            len(fused),
            len(dense_hits),
            len(sparse_hits),
            self._rrf_k,
        )
        return fused

    def _ensure_bm25(self) -> BM25Index:
        total = self._index.count()
        if self._bm25 is not None and self._bm25.corpus_size == total:
            return self._bm25
        self._bm25 = self._build_bm25()
        return self._bm25

    def _build_bm25(self) -> BM25Index:
        stored = self._index.list_chunks()
        if not stored:
            raise RetrievalError(
                f"Cannot build BM25 over an empty collection {self._index.collection_name}"
            )
        return BM25Index(stored, k1=self._bm25_k1, b=self._bm25_b)


def _valid_bm25_params(*, k1: float, b: float) -> bool:
    return math.isfinite(k1) and k1 > 0 and math.isfinite(b) and 0.0 <= b <= 1.0


def _hit_payload(
    dense: SearchHit | None,
    sparse: SparseHit | None,
    *,
    chunk_id: str,
) -> tuple[str, dict[str, str]]:
    if dense is not None:
        return dense.text, dict(dense.metadata)
    if sparse is not None:
        return sparse.text, dict(sparse.metadata)
    raise RetrievalError(f"Fused chunk {chunk_id} is missing from both rankings")
