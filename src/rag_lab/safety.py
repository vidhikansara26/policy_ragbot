"""Screen a reranked window before it is shown to the generator.

``search()`` still returns low-scoring and ``status=legacy`` hits. This module
drops scores below :data:`~rag_lab.config.MIN_RERANK_SCORE` and, for a
one-token query, drops chunks that do not contain that token.

The floor has one exception. A retired passage often scores higher than the
policy that replaced it, because the retired text still answers the question
literally while the current text says the entitlement no longer exists. Letting
the floor keep the legacy hit and drop its current twin would leave a window
that cannot state current policy and cannot show the versions disagree, so a
surviving legacy hit re-admits the best current hit from the same document.
This module does not read or write the index.
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
    must contain that token. Longer questions skip the token check. A surviving
    legacy hit then re-admits the best current hit of the same document, even
    below the floor. An empty survivor list abstains before that, so the
    exception cannot resurrect a window the floor rejected outright. The index
    is not modified.

    Raises:
        GenerationError: If a hit score is not a finite number.
    """
    checked = tuple(_require_finite(hit) for hit in hits)
    scored = tuple(hit for hit in checked if hit.score >= MIN_RERANK_SCORE)
    kept = _apply_lexical_gate(question, scored)
    if not kept:
        return ScreenedContext(hits=(), abstain=True, has_legacy_conflict=False)
    ordered = _current_then_legacy(_with_current_companions(question, kept, checked))
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


def _with_current_companions(
    question: str,
    kept: Sequence[RerankedHit],
    candidates: Sequence[RerankedHit],
) -> tuple[RerankedHit, ...]:
    """Re-admit the current counterpart of every surviving legacy hit.

    A companion is drawn from the pre-floor window and must still satisfy the
    lexical gate, so this widens the window by version lineage only, never by
    relevance. Documents that already have a current hit are untouched.
    """
    orphans = _legacy_only_docs(kept)
    if not orphans:
        return tuple(kept)
    kept_ids = {hit.chunk_id for hit in kept}
    gated = _apply_lexical_gate(question, candidates)
    pool = [hit for hit in gated if hit.chunk_id not in kept_ids]
    companions: list[RerankedHit] = []
    for doc_name in orphans:
        twins = [hit for hit in pool if _doc_status(hit) == (doc_name, _STATUS_CURRENT)]
        if twins:
            companions.append(max(twins, key=lambda hit: hit.score))
    return (*kept, *companions)


def _legacy_only_docs(hits: Sequence[RerankedHit]) -> list[str]:
    """Document names present as legacy but not as current, in first-seen order."""
    statuses = _statuses_by_doc(hits)
    return [
        doc_name
        for doc_name, seen in statuses.items()
        if _STATUS_LEGACY in seen and _STATUS_CURRENT not in seen
    ]


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
    return any(
        _STATUS_CURRENT in statuses and _STATUS_LEGACY in statuses
        for statuses in _statuses_by_doc(hits).values()
    )


def _statuses_by_doc(hits: Sequence[RerankedHit]) -> dict[str, set[str]]:
    """Map each document name to the statuses it appears under, in first-seen order."""
    statuses: dict[str, set[str]] = {}
    for hit in hits:
        pair = _doc_status(hit)
        if pair is None:
            continue
        doc_name, status = pair
        statuses.setdefault(doc_name, set()).add(status)
    return statuses


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
