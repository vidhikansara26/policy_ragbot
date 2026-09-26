"""Two-question diagnosis of the planted stale Privilege Leave (PTO) clause.

Every claim in ``docs/data-quality-diagnosis.md`` is pinned here. The rows are
deterministic: chunking is pure, and the retrieval window is scripted with the
real chunk bodies and the cross-encoder scores captured from production. No
model weights are downloaded and no Chroma directory is created, so the suite
runs under ``pytest -m "not integration"`` in CI.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_lab import config
from rag_lab.chunking import Chunk, chunk_corpus
from rag_lab.corpus import load_corpus
from rag_lab.eval import RetrievalLabel, bind_gold, evaluate_retrieval, evaluation_cases
from rag_lab.generate import generate_answer, render_answer
from rag_lab.rerank import RerankedHit
from rag_lab.safety import screen_hits

_PTO_QUESTION = "How many Privilege Leave / PTO days do I get?"
_LEGACY_SPAN = f"{config.LEGACY_PTO_DAYS} days of Privilege Leave (PTO)"
_CURRENT_SPAN = "does not set a numeric Privilege Leave (PTO) entitlement"
_DIAGNOSIS_DOC = config.PROJECT_ROOT / "docs" / "data-quality-diagnosis.md"

# Cross-encoder scores for _PTO_QUESTION on commit 9c6d8b7. The legacy clause
# outranks its current twin because only the retired document states a day
# count. The current twin is below MIN_RERANK_SCORE.
_LEGACY_SCORE = 6.406
_SUPPLIER_SCORE = 0.106
_CURRENT_SCORE = -1.023


class _ScriptedWindow:
    """Returns one fixed reranked window for the Privilege Leave question."""

    def __init__(self, hits: Sequence[RerankedHit]) -> None:
        self._hits = list(hits)

    def search(self, query: str, *, k: int) -> list[RerankedHit]:
        """Return the scripted window truncated to ``k``.

        Raises:
            AssertionError: If asked for a question this window does not cover.
        """
        assert query == _PTO_QUESTION
        return self._hits[:k]


class _FakeGenerator:
    """Returns a fixed completion and records how often it was called."""

    def __init__(self, completion: str) -> None:
        self.completion = completion
        self.calls = 0

    def complete(self, prompt: str) -> str:
        """Return the canned completion."""
        del prompt
        self.calls += 1
        return self.completion


def _chunk_holding(span: str) -> Chunk:
    holders = [chunk for chunk in chunk_corpus(load_corpus()) if span in chunk.text]
    assert len(holders) == 1, f"{span!r} must sit in exactly one chunk, found {len(holders)}"
    return holders[0]


def _hit(chunk: Chunk, score: float) -> RerankedHit:
    return RerankedHit(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        score=score,
        metadata=dict(chunk.metadata),
        rrf_score=0.03,
        dense_rank=1,
        bm25_rank=1,
        dense_distance=0.4,
        bm25_score=1.0,
    )


@pytest.fixture
def window() -> list[RerankedHit]:
    """The production top-3 for the Privilege Leave question, in rank order."""
    legacy = _hit(_chunk_holding(_LEGACY_SPAN), _LEGACY_SCORE)
    supplier = _hit(_chunk_holding("adequate rest periods and parental leave"), _SUPPLIER_SCORE)
    current = _hit(_chunk_holding(_CURRENT_SPAN), _CURRENT_SCORE)
    return [legacy, supplier, current]


def test_conflict_lives_in_the_source_files_not_the_retriever() -> None:
    """Root cause: two indexed files disagree before any retrieval runs."""
    by_version = {
        document.version: document
        for document in load_corpus()
        if document.doc_name == config.PLANTED_DOC_NAME
    }
    legacy = by_version[config.LEGACY_POLICY_VERSION]
    current = by_version[config.CURRENT_POLICY_VERSION]

    assert legacy.status == "legacy"
    assert current.status == "current"
    assert _LEGACY_SPAN in legacy.body
    assert _CURRENT_SPAN in current.body
    assert _LEGACY_SPAN not in current.body


def test_question_one_retrieval_surfaces_both_versions(window: list[RerankedHit]) -> None:
    """Answer to question 1 is Yes: v1 and v2 are both inside the scored window."""
    gold = [
        passage
        for passage in bind_gold(evaluation_cases(), load_corpus())
        if passage.query_id == "pto_privilege_leave"
    ]
    report = evaluate_retrieval(_ScriptedWindow(window), gold, k=config.EVAL_K)
    result = report.by_id("pto_privilege_leave")

    assert result.recall_at_k == 1.0
    assert result.retrieval_label is RetrievalLabel.DATA_QUALITY_FIXTURE
    assert "not a failed retrieval" in result.note
    versions = result.versions_for(config.PLANTED_DOC_NAME)
    assert config.LEGACY_POLICY_VERSION in versions
    assert config.CURRENT_POLICY_VERSION in versions


def test_question_two_extractive_answer_uses_the_legacy_source(
    window: list[RerankedHit],
) -> None:
    """Answer to question 2 is No: the cited passage is the retired v1 clause."""
    gold = [
        passage
        for passage in bind_gold(evaluation_cases(), load_corpus())
        if passage.query_id == "pto_privilege_leave"
    ]
    answer = evaluate_retrieval(
        _ScriptedWindow(window),
        gold,
        k=config.EVAL_K,
    ).by_id("pto_privilege_leave").answer

    assert answer.doc_name == config.PLANTED_DOC_NAME
    assert answer.version == config.LEGACY_POLICY_VERSION
    assert answer.status == "legacy"
    assert _LEGACY_SPAN in answer.text


def test_remediation_leak_guard_never_publishes_the_legacy_day_count(
    window: list[RerankedHit],
) -> None:
    """A model that repeats the retired number is not allowed to publish it."""
    generator = _FakeGenerator(
        json.dumps(
            {
                "grounded": True,
                "answer": f"You receive {config.LEGACY_PTO_DAYS} days of Privilege Leave.",
            }
        )
    )
    published = render_answer(generate_answer(_PTO_QUESTION, window, generator))

    assert generator.calls == 1
    assert f"{config.LEGACY_PTO_DAYS} days" not in published
    assert config.PLANTED_DOC_NAME in published


def test_companion_rule_keeps_the_current_twin_below_the_score_floor(
    window: list[RerankedHit],
) -> None:
    """Remediation step 2: the floor no longer hides the policy that replaced v1.

    The cross-encoder scores the retired clause above its current counterpart,
    because only the retired text states a day count. Dropping the current twin
    on score alone would leave a window that cannot state current policy.
    """
    assert _CURRENT_SCORE < config.MIN_RERANK_SCORE <= _LEGACY_SCORE
    screened = screen_hits(_PTO_QUESTION, window)

    kept = [(hit.metadata["version"], hit.metadata["status"]) for hit in screened.hits]
    assert (config.CURRENT_POLICY_VERSION, "current") in kept
    assert (config.LEGACY_POLICY_VERSION, "legacy") in kept
    assert screened.has_legacy_conflict is True
    statuses = [hit.metadata["status"] for hit in screened.hits]
    assert statuses.index("current") < statuses.index("legacy")


def test_remediation_marks_the_legacy_citation_as_a_conflict(
    window: list[RerankedHit],
) -> None:
    """Remediation step 4: v1 is cited beside v2 and labelled, not hidden."""
    published = "The current Human Rights Policy does not set a numeric PTO entitlement."
    generator = _FakeGenerator(json.dumps({"grounded": True, "answer": published}))
    answer = generate_answer(_PTO_QUESTION, window, generator)
    rendered = render_answer(answer)

    assert answer.abstained is False
    assert answer.has_legacy_conflict is True
    assert f"{config.LEGACY_PTO_DAYS} days" not in rendered
    assert "(legacy conflict)" in rendered
    legacy = next(
        item for item in answer.citations if item.version == config.LEGACY_POLICY_VERSION
    )
    assert legacy.legacy_conflict is True
    assert any(
        item.version == config.CURRENT_POLICY_VERSION and not item.legacy_conflict
        for item in answer.citations
    )


def test_diagnosis_document_answers_both_questions() -> None:
    """The written diagnosis states both verdicts and traces them to source data."""
    assert _DIAGNOSIS_DOC.is_file()
    text = _DIAGNOSIS_DOC.read_text(encoding="utf-8")

    assert "Question 1 — did retrieval find the right documents? **Yes.**" in text
    assert "Question 2 — did the system use the right one? **No.**" in text
    assert "## Root cause: source data" in text
    assert "## Remediation" in text
    assert Path("data/raw/human_rights_policy_v1.md").as_posix() in text
    assert Path("data/raw/human_rights_policy_v2.md").as_posix() in text
