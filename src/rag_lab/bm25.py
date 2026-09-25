"""Okapi BM25 over policy chunks.

Sparse scores are not on the same scale as MiniLM cosine distance. Hybrid
search therefore treats this module's ranked list as an input to Reciprocal
Rank Fusion, rather than adding the raw BM25 number to a dense similarity.

Tokens are lowercase alphanumeric runs. Numbers such as ``15`` are kept, and
there is no stopword list: ``not`` and ``days`` distinguish the stale Privilege
Leave clause from the current policy's "does not set a numeric entitlement"
wording. ``k1`` and ``b`` come from ``rag_lab.config``.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_lab.config import BM25_B, BM25_K1
from rag_lab.exceptions import IndexingError, RetrievalError

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


class IndexableChunk(Protocol):
    """Fields BM25 needs from either a ``Chunk`` or a stored Chroma row."""

    @property
    def chunk_id(self) -> str:
        """Stable chunk identifier."""
        ...

    @property
    def text(self) -> str:
        """Chunk body that is tokenized into the postings."""
        ...

    @property
    def metadata(self) -> dict[str, str]:
        """Attribution metadata copied onto each hit."""
        ...


@dataclass(frozen=True)
class SparseHit:
    """One BM25 hit. ``score`` is higher-is-better and is not a distance."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, str]


@dataclass(frozen=True)
class _Bm25Doc:
    chunk_id: str
    text: str
    metadata: dict[str, str]
    term_freqs: dict[str, int]
    length: int


def tokenize(text: str) -> list[str]:
    """Split ``text`` into lowercase alphanumeric tokens.

    Punctuation is discarded. ``Privilege Leave / PTO`` becomes
    ``privilege``, ``leave``, ``pto``.
    """
    return _TOKEN.findall(text.lower())


class BM25Index:
    """In-memory Okapi BM25 index. Rebuild it from the persisted chunks."""

    def __init__(
        self,
        chunks: Sequence[IndexableChunk],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        """Index ``chunks`` for lexical search. Does not drop ``status=legacy``.

        Raises:
            IndexingError: If parameters are invalid, ``chunks`` is empty, or a
                row has a blank id, blank text, or no alphanumeric tokens.
        """
        if not math.isfinite(k1) or k1 <= 0:
            raise IndexingError(f"Invalid BM25 k1={k1}")
        if not math.isfinite(b) or not 0.0 <= b <= 1.0:
            raise IndexingError(f"Invalid BM25 b={b}")
        if not chunks:
            raise IndexingError("Cannot build BM25 over an empty chunk list")
        self._k1 = k1
        self._b = b
        self._docs = _build_docs(chunks)
        self._idf = _idf_weights(self._docs)
        total_length = sum(doc.length for doc in self._docs)
        self._avgdl = total_length / len(self._docs)
        logger.info(
            "Built BM25 over %s chunks (vocab=%s, avgdl=%.1f, k1=%s, b=%s)",
            len(self._docs),
            len(self._idf),
            self._avgdl,
            k1,
            b,
        )

    @property
    def corpus_size(self) -> int:
        """Number of chunks in the sparse index."""
        return len(self._docs)

    def search(self, query: str, *, k: int) -> list[SparseHit]:
        """Return up to ``k`` chunks with a positive BM25 score, best first.

        Chunks that share no query token are omitted. Ties break on
        ``chunk_id`` so the rank order is deterministic. No metadata filter
        is applied, so a legacy Human Rights chunk can outrank the current
        policy when the query terms actually occur there.

        Raises:
            RetrievalError: If ``query`` is empty, has no tokens, or ``k`` is invalid.
        """
        if k <= 0:
            raise RetrievalError(f"Invalid k={k}")
        if not query.strip():
            raise RetrievalError("Query text is empty")
        tokens = tokenize(query)
        if not tokens:
            raise RetrievalError("Query has no searchable tokens")
        scored: list[tuple[float, _Bm25Doc]] = []
        for doc in self._docs:
            score = self._score_document(doc, tokens)
            if score > 0.0:
                scored.append((score, doc))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        hits: list[SparseHit] = []
        for score, doc in scored[:k]:
            hits.append(
                SparseHit(
                    chunk_id=doc.chunk_id,
                    text=doc.text,
                    score=score,
                    metadata=dict(doc.metadata),
                )
            )
        logger.info("BM25 returned %s hits for a %s-token query", len(hits), len(tokens))
        return hits

    def _score_document(self, doc: _Bm25Doc, query_tokens: list[str]) -> float:
        """Okapi score. Query-term repeats add another saturated tf contribution."""
        score = 0.0
        length_norm = 1.0 - self._b + self._b * (doc.length / self._avgdl)
        for token in query_tokens:
            idf = self._idf.get(token)
            if idf is None:
                continue
            tf = doc.term_freqs.get(token, 0)
            if tf == 0:
                continue
            numerator = tf * (self._k1 + 1.0)
            denominator = tf + self._k1 * length_norm
            score += idf * (numerator / denominator)
        return score


def _build_docs(chunks: Sequence[IndexableChunk]) -> list[_Bm25Doc]:
    seen: set[str] = set()
    docs: list[_Bm25Doc] = []
    for chunk in chunks:
        chunk_id = chunk.chunk_id
        if not chunk_id.strip():
            raise IndexingError("Chunk is missing chunk_id")
        if chunk_id in seen:
            raise IndexingError(f"Duplicate chunk_id in BM25 index: {chunk_id}")
        seen.add(chunk_id)
        if not chunk.text.strip():
            raise IndexingError(f"Refusing to index an empty chunk body: {chunk_id}")
        tokens = tokenize(chunk.text)
        if not tokens:
            raise IndexingError(f"Chunk {chunk_id} has no alphanumeric tokens")
        docs.append(
            _Bm25Doc(
                chunk_id=chunk_id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                term_freqs=dict(Counter(tokens)),
                length=len(tokens),
            )
        )
    return docs


def _idf_weights(docs: list[_Bm25Doc]) -> dict[str, float]:
    """Lucene-style idf: ``log(1 + (N - df + 0.5) / (df + 0.5))``.

    The ``+ 1`` inside the log keeps weights positive when a term appears in
    every chunk, which the plain Robertson idf does not.
    """
    doc_freq: dict[str, int] = {}
    for doc in docs:
        for token in doc.term_freqs:
            doc_freq[token] = doc_freq.get(token, 0) + 1
    n_docs = len(docs)
    return {
        token: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
        for token, freq in doc_freq.items()
    }
