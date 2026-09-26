"""Screen a reranked window before it is shown to the generator.

``search()`` still returns low-scoring and ``status=legacy`` hits. This module
drops scores below :data:`~rag_lab.config.MIN_RERANK_SCORE` and, for a
one-token query, drops chunks that do not contain that token. It does not
read or write the index.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag_lab.config import MIN_RERANK_SCORE, SHORT_QUERY_MAX_TOKENS
from rag_lab.exceptions import GenerationError
from rag_lab.rerank import RerankedHit

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_STATUS_CURRENT = "current"
_STATUS_LEGACY = "legacy"


@dataclass(frozen=True)
class ScreenedContext:
    """Hits that may enter the generator prompt.

    ``hits`` is empty when ``abstain`` is true. Current rows keep their
    incoming order and come before legacy rows. ``has_legacy_conflict`` is
    true when one ``doc_name`` still has both ``current`` and ``legacy``.
    """

    hits: tuple[RerankedHit, ...]
    abstain: bool
    has_legacy_conflict: bool


def screen_hits(question: str, hits: Sequence[RerankedHit]) -> ScreenedContext:
    """Drop irrelevant hits and put current policy ahead of legacy policy.

    A score below :data:`~rag_lab.config.MIN_RERANK_SCORE` is removed. A score
    equal to the threshold stays. When ``question`` has fewer than
    :data:`~rag_lab.config.SHORT_QUERY_MAX_TOKENS` alphanumeric tokens, a hit
    must contain that token. Longer questions skip the token check. An empty
    survivor list abstains. The index is not modified.

    Raises:
        GenerationError: If a hit score is not a finite number.
    """
    checked = tuple(_require_finite(hit) for hit in hits)
    scored = tuple(hit for hit in checked if hit.score >= MIN_RERANK_SCORE)
    kept = _apply_lexical_gate(question, scored)
    if not kept:
        return ScreenedContext(hits=(), abstain=True, has_legacy_conflict=False)
    ordered = _current_then_legacy(kept)
    return ScreenedContext(
        hits=ordered,
        abstain=False,
        has_legacy_conflict=_has_legacy_conflict(ordered),
    )


def _require_finite(hit: RerankedHit) -> RerankedHit:
    if not math.isfinite(hit.score):
        raise GenerationError(f"Chunk {hit.chunk_id} has a non-finite rerank score")
    return hit


def _apply_lexical_gate(question: str, hits: Sequence[RerankedHit]) -> tuple[RerankedHit, ...]:
    tokens = _TOKEN.findall(question)
    if len(tokens) >= SHORT_QUERY_MAX_TOKENS:
        return tuple(hits)
    if len(tokens) != 1:
        return ()
    token = tokens[0]
    pattern = re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
    return tuple(hit for hit in hits if pattern.search(hit.text) is not None)


def _current_then_legacy(hits: Sequence[RerankedHit]) -> tuple[RerankedHit, ...]:
    """Stable order: current, then unrecognized status, then legacy."""
    return tuple(sorted(hits, key=_status_rank))


def _status_rank(hit: RerankedHit) -> int:
    status = _status(hit)
    if status == _STATUS_CURRENT:
        return 0
    if status == _STATUS_LEGACY:
        return 2
    return 1


def _has_legacy_conflict(hits: Sequence[RerankedHit]) -> bool:
    statuses_by_doc: dict[str, set[str]] = {}
    for hit in hits:
        pair = _doc_status(hit)
        if pair is None:
            continue
        doc_name, status = pair
        statuses_by_doc.setdefault(doc_name, set()).add(status)
    return any(
        _STATUS_CURRENT in statuses and _STATUS_LEGACY in statuses
        for statuses in statuses_by_doc.values()
    )


def _doc_status(hit: RerankedHit) -> tuple[str, str] | None:
    doc_name = hit.metadata.get("doc_name", "")
    status = hit.metadata.get("status", "")
    if not isinstance(doc_name, str) or not isinstance(status, str):
        return None
    doc_name = doc_name.strip()
    status = status.strip()
    if not doc_name or status not in {_STATUS_CURRENT, _STATUS_LEGACY}:
        return None
    return doc_name, status


def _status(hit: RerankedHit) -> str:
    pair = _doc_status(hit)
    if pair is None:
        return ""
    return pair[1]
