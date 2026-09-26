"""Grounded answer generation from a reranked hit window.

The model sees only the chunks ``search()`` already returned, after
:func:`rag_lab.safety.screen_hits` drops scores below the relevance floor and
puts current passages ahead of legacy ones. Each passage is tagged with
``doc_name``, ``section``, ``version``, and ``status``. A legacy passage is
also labeled ``LEGACY``. Python builds citations and decides what may ship.
``status=legacy`` is not filtered at ingest; a legacy row stays in the prompt
so a conflict is visible, and it is never published as current policy.

The planted ``LEGACY_PTO_DAYS`` count is not a current entitlement. When the
current sibling does not contain it, that number is dropped. A legacy-only
window never calls the model.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rag_lab.config import (
    LEGACY_PTO_DAYS,
    LLM_API_KEY_ENV,
    LLM_MODEL_ENV,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)
from rag_lab.exceptions import GenerationError
from rag_lab.rerank import RerankedHit
from rag_lab.safety import ScreenedContext, screen_hits

logger = logging.getLogger(__name__)

ABSTAIN_TEXT: str = "I cannot answer from the retrieved policies."
_CURRENT_PTO_TEXT: str = "The current Human Rights Policy does not specify a numeric PTO allowance."

_CITATION_FIELDS: tuple[str, ...] = ("doc_name", "section", "version", "status")
_SECTION_NUMBER_PREFIX = re.compile(r"^\d+\.\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_LONG_NUMBER = re.compile(r"\d{7,}")

_STATUS_CURRENT = "current"
_STATUS_LEGACY = "legacy"


class TextGenerator(Protocol):
    """Completes a prompt string. Used by :func:`generate_answer`."""

    def complete(self, prompt: str) -> str:
        """Return model text for ``prompt``.

        Raises:
            GenerationError: If the prompt is empty or the provider call fails.
        """
        ...


@dataclass(frozen=True)
class Citation:
    """One source line. ``section`` is the chunk metadata string, unchanged."""

    doc_name: str
    section: str
    version: str
    status: str
    legacy_conflict: bool


@dataclass(frozen=True)
class GeneratedAnswer:
    """A grounded answer, or an abstention, plus the citations Python built."""

    text: str
    citations: tuple[Citation, ...]
    abstained: bool
    has_legacy_conflict: bool


@dataclass(frozen=True)
class _CitedHit:
    """One validated hit. ``section_key`` groups numbered siblings."""

    hit: RerankedHit
    doc_name: str
    section: str
    section_key: str
    version: str
    status: str


@dataclass(frozen=True)
class _ConflictGroup:
    """Hits that share ``(doc_name, section_key)``."""

    key: tuple[str, str]
    hits: tuple[_CitedHit, ...]
    has_current: bool
    has_legacy: bool

    @property
    def is_conflict(self) -> bool:
        return self.has_current and self.has_legacy


class OpenAIChatGenerator:
    """Chat completions through an already constructed OpenAI client.

    The ``openai`` package is imported in :func:`generator_from_env`, not here,
    so unit tests can pass a fake :class:`TextGenerator` without the SDK.
    """

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        temperature: float = LLM_TEMPERATURE,
        timeout_seconds: float = LLM_TIMEOUT_SECONDS,
    ) -> None:
        """Bind a client and model. No network call until ``complete``.

        Raises:
            GenerationError: If ``model`` is blank or ``timeout_seconds`` is invalid.
        """
        if not model.strip():
            raise GenerationError("LLM model is empty")
        if timeout_seconds <= 0:
            raise GenerationError(f"Invalid LLM timeout {timeout_seconds}")
        self._client = client
        self._model = model.strip()
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds

    def complete(self, prompt: str) -> str:
        """POST a chat completion and return the assistant message text.

        Raises:
            GenerationError: If ``prompt`` is empty, the call fails, or the
                response has no message content.
        """
        if not prompt.strip():
            raise GenerationError("Generation prompt is empty")
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                timeout=self._timeout_seconds,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise GenerationError(f"LLM request failed: {exc}") from exc
        return _message_content(response)


def generator_from_env() -> TextGenerator:
    """Build an OpenAI chat generator from the environment.

    ``OPENAI_API_KEY`` and ``RAG_LAB_LLM_MODEL`` are required. The key is never
    hard-coded and is not logged. The ``openai`` package is imported only on
    this path.

    Raises:
        GenerationError: If the key or model is missing, or ``openai`` is not
            installed. Raised before any HTTP call.
    """
    api_key = os.environ.get(LLM_API_KEY_ENV, "").strip()
    model = os.environ.get(LLM_MODEL_ENV, "").strip()
    if not api_key:
        raise GenerationError(f"{LLM_API_KEY_ENV} is not set")
    if not model:
        raise GenerationError(f"{LLM_MODEL_ENV} is not set")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise GenerationError(
            'The openai package is not installed. Install with pip install -e ".[llm]".'
        ) from exc

    logger.info("LLM provider=openai model=%s", model)
    client = OpenAI(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)
    return OpenAIChatGenerator(client, model=model)


def generate_answer(
    question: str,
    hits: Sequence[RerankedHit],
    generator: TextGenerator,
) -> GeneratedAnswer:
    """Answer ``question`` from ``hits`` only.

    ``hits`` is the reranked ``search()`` window. Scores below the relevance
    floor, and one-token queries whose token is missing from a chunk, are
    dropped before the prompt is built. If nothing remains, or every survivor
    is legacy, the call abstains without calling ``generator``. When a current
    passage is present, the model returns ``{"grounded": bool, "answer": str}``.
    Citations are built here. A legacy citation is marked as a conflict when
    the same policy section also has a current hit, and on a legacy-only window.

    Raises:
        GenerationError: If the question is empty, a hit score is not finite,
            a hit is missing citation metadata, the model returns unparseable
            JSON, or ``generator`` fails.
    """
    if not question.strip():
        raise GenerationError("Question text is empty")
    screened = screen_hits(question, hits)
    if screened.abstain:
        return _abstain(())
    cited = tuple(_validate_hit(hit) for hit in screened.hits)
    if not cited:
        return _abstain(())

    groups = _group_hits(cited)
    current = tuple(item for item in cited if item.status == _STATUS_CURRENT)
    legacy = tuple(item for item in cited if item.status == _STATUS_LEGACY)
    legacy_only = not current
    citations = _citations(cited, groups, legacy_only=legacy_only)

    conflict = _window_has_legacy_conflict(screened, citations)
    if legacy_only:
        return _abstain(citations, has_legacy_conflict=conflict)

    prompt = _build_prompt(question.strip(), cited)
    raw = generator.complete(prompt)
    grounded, answer = _parse_model_json(raw)
    if not grounded:
        return _abstain(citations, has_legacy_conflict=conflict)

    prose = _publishable_prose(answer, current=current, legacy=legacy)
    if prose is None:
        return _abstain(citations, has_legacy_conflict=conflict)
    return GeneratedAnswer(
        text=prose,
        citations=citations,
        abstained=False,
        has_legacy_conflict=conflict,
    )


def render_answer(answer: GeneratedAnswer) -> str:
    """Format ``answer`` as prose plus a Sources block.

    Legacy conflict citations keep the metadata section string and a
    ``(legacy conflict)`` suffix.
    """
    lines = [answer.text.rstrip(), "Sources:"]
    if not answer.citations:
        lines.append("- (none)")
    else:
        for citation in answer.citations:
            suffix = " (legacy conflict)" if citation.legacy_conflict else ""
            lines.append(f"- {citation.doc_name}, {citation.section}, v{citation.version}{suffix}")
    return "\n".join(lines) + "\n"


def _abstain(
    citations: tuple[Citation, ...],
    *,
    has_legacy_conflict: bool | None = None,
) -> GeneratedAnswer:
    flagged = (
        any(item.legacy_conflict for item in citations)
        if has_legacy_conflict is None
        else has_legacy_conflict
    )
    return GeneratedAnswer(
        text=ABSTAIN_TEXT,
        citations=citations,
        abstained=True,
        has_legacy_conflict=flagged,
    )


def _window_has_legacy_conflict(
    screened: ScreenedContext,
    citations: tuple[Citation, ...],
) -> bool:
    """Document-level screen flag, or a legacy citation already marked in conflict."""
    return screened.has_legacy_conflict or any(item.legacy_conflict for item in citations)


def _validate_hit(hit: RerankedHit) -> _CitedHit:
    values: dict[str, str] = {}
    for field in _CITATION_FIELDS:
        raw = hit.metadata.get(field, "")
        if not isinstance(raw, str) or not raw.strip():
            raise GenerationError(f"Chunk {hit.chunk_id} is missing {field}")
        values[field] = raw.strip()
    status = values["status"]
    if status not in {_STATUS_CURRENT, _STATUS_LEGACY}:
        raise GenerationError(f"Chunk {hit.chunk_id} has unsupported status {status!r}")
    if not hit.text.strip():
        raise GenerationError(f"Chunk {hit.chunk_id} has empty text")
    section = values["section"]
    return _CitedHit(
        hit=hit,
        doc_name=values["doc_name"],
        section=section,
        section_key=_normalize_section(section),
        version=values["version"],
        status=status,
    )


def _normalize_section(section: str) -> str:
    """Drop a leading ``N. `` so v1 ``3. Fair Wages…`` matches v2 ``5. Fair Wages…``."""
    return _SECTION_NUMBER_PREFIX.sub("", section.strip())


def _group_hits(cited: Sequence[_CitedHit]) -> list[_ConflictGroup]:
    order: list[tuple[str, str]] = []
    buckets: dict[tuple[str, str], list[_CitedHit]] = {}
    for item in cited:
        key = (item.doc_name, item.section_key)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(item)
    groups: list[_ConflictGroup] = []
    for key in order:
        members = tuple(buckets[key])
        groups.append(
            _ConflictGroup(
                key=key,
                hits=members,
                has_current=any(member.status == _STATUS_CURRENT for member in members),
                has_legacy=any(member.status == _STATUS_LEGACY for member in members),
            )
        )
    return groups


def _citations(
    cited: Sequence[_CitedHit],
    groups: Sequence[_ConflictGroup],
    *,
    legacy_only: bool,
) -> tuple[Citation, ...]:
    conflict_keys = {group.key for group in groups if group.is_conflict}
    seen: set[tuple[str, str, str, str]] = set()
    citations: list[Citation] = []
    for item in _citation_order(cited):
        identity = (item.doc_name, item.section, item.version, item.status)
        if identity in seen:
            continue
        seen.add(identity)
        mark_conflict = item.status == _STATUS_LEGACY and (
            (item.doc_name, item.section_key) in conflict_keys or legacy_only
        )
        citations.append(
            Citation(
                doc_name=item.doc_name,
                section=item.section,
                version=item.version,
                status=item.status,
                legacy_conflict=mark_conflict,
            )
        )
    return tuple(citations)


def _citation_order(cited: Sequence[_CitedHit]) -> list[_CitedHit]:
    """Current hits first (window order), then legacy hits (window order)."""
    current = [item for item in cited if item.status == _STATUS_CURRENT]
    legacy = [item for item in cited if item.status == _STATUS_LEGACY]
    return [*current, *legacy]


def _build_prompt(question: str, cited: Sequence[_CitedHit]) -> str:
    lines = [
        "You answer enterprise policy questions using ONLY the passages below.",
        "A passage with status=legacy is retired. Never treat it as current policy.",
        "If a current passage and a legacy passage of the same policy disagree, "
        "answer only from the current passage.",
        "Never invent day counts, emails, or clauses that are not in a status=current passage.",
        'Respond with a single JSON object: {"grounded": <bool>, "answer": "<string>"}.',
        "Set grounded to false when no current passage supports the question.",
        "Do not include a Sources block; citations are added separately.",
        "",
        f"Question: {question}",
        "",
        "Passages:",
    ]
    for index, item in enumerate(cited, start=1):
        header = (
            f"[{index}] doc_name={item.doc_name} | section={item.section} | "
            f"version={item.version} | status={item.status}"
        )
        if item.status == _STATUS_LEGACY:
            header += " | label=LEGACY"
        lines.append(header)
        lines.append(item.hit.text.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _parse_model_json(raw: str) -> tuple[bool, str]:
    text = raw.strip()
    if not text:
        raise GenerationError("Model returned an empty completion")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        fenced = _strip_json_fence(text)
        try:
            payload = json.loads(fenced)
        except json.JSONDecodeError as exc:
            raise GenerationError("Model returned unparseable JSON") from exc
    if not isinstance(payload, dict):
        raise GenerationError("Model JSON must be an object")
    if "grounded" not in payload or "answer" not in payload:
        raise GenerationError("Model JSON must include grounded and answer")
    grounded = payload["grounded"]
    answer = payload["answer"]
    if not isinstance(grounded, bool):
        raise GenerationError("Model JSON grounded must be a boolean")
    if not isinstance(answer, str):
        raise GenerationError("Model JSON answer must be a string")
    return grounded, answer.strip()


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _publishable_prose(
    answer: str,
    *,
    current: Sequence[_CitedHit],
    legacy: Sequence[_CitedHit],
) -> str | None:
    """Return prose that current chunks support, or None to abstain.

    ``LEGACY_PTO_DAYS`` is replaced with the current-policy sentence when the
    window is the Privilege Leave conflict and no current chunk contains that
    count. Any other unsupported token or legacy-only sentence is not published.
    """
    cleaned = answer.strip()
    if not cleaned:
        return None

    current_text = "\n".join(item.hit.text for item in current)
    legacy_text = "\n".join(item.hit.text for item in legacy)
    day_phrase = f"{LEGACY_PTO_DAYS} days"
    pto_conflict = bool(current) and bool(legacy) and _mentions_pto(current_text, legacy_text)

    if day_phrase in cleaned and day_phrase not in current_text:
        logger.warning("Dropped a legacy day count that current policy does not state")
        if pto_conflict:
            return _CURRENT_PTO_TEXT
        return None

    if _has_unsupported_token(cleaned, current_text):
        logger.warning("Dropped an answer token that no current passage contains")
        return None

    if _contains_legacy_only_sentence(cleaned, current_text=current_text, legacy_text=legacy_text):
        logger.warning("Dropped a sentence that only a legacy passage supports")
        if pto_conflict:
            return _CURRENT_PTO_TEXT
        return None

    return cleaned


def _mentions_pto(current_text: str, legacy_text: str) -> bool:
    needle = "Privilege Leave"
    return needle in current_text or needle in legacy_text


def _has_unsupported_token(answer: str, current_text: str) -> bool:
    if any(email not in current_text for email in _EMAIL.findall(answer)):
        return True
    return any(number not in current_text for number in _LONG_NUMBER.findall(answer))


def _contains_legacy_only_sentence(
    answer: str,
    *,
    current_text: str,
    legacy_text: str,
) -> bool:
    if not legacy_text.strip():
        return False
    for sentence in _SENTENCE_SPLIT.split(answer):
        piece = sentence.strip().strip("\"'")
        if len(piece) < 12:
            continue
        if piece in legacy_text and piece not in current_text:
            return True
    return False


def _message_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise GenerationError("LLM response has no choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise GenerationError("LLM response has empty content")
    return content
