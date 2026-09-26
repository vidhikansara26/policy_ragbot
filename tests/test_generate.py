"""Deterministic answer generation from a reranked hit window.

Unit tests inject a fake ``TextGenerator``. The live OpenAI path is marked
``integration`` and skips when ``OPENAI_API_KEY`` is unset.
"""

from __future__ import annotations

import json
import os

import pytest

from rag_lab import config
from rag_lab.exceptions import GenerationError
from rag_lab.generate import (
    ABSTAIN_TEXT,
    generate_answer,
    generator_from_env,
    render_answer,
)
from rag_lab.rerank import RerankedHit

_CURRENT_PTO = (
    "This current policy does not set a numeric Privilege Leave (PTO) entitlement. "
    "Leave, rest days, overtime, and other statutory entitlements follow the "
    "applicable local employment law and the employee's service conditions."
)
_LEGACY_PTO = (
    f"Full-time employees are entitled to {config.LEGACY_PTO_DAYS} days of "
    "Privilege Leave (PTO) per calendar year. Privilege Leave accrues annually "
    "on 1 January."
)
_PTO_QUESTION = "How many Privilege Leave / PTO days do I get?"
_CURRENT_PTO_SENTENCE = "The current Human Rights Policy does not specify a numeric PTO allowance."
_DECOY = "Decoy passage about board sitting fees that must not appear in the prompt."
_POSH_TEXT = "The Sexual Harassment Redressal Committee email id is shrc@coforge.com."


class _FakeGenerator:
    """Records prompts and returns a fixed completion string."""

    def __init__(self, completion: str) -> None:
        self.completion = completion
        self.prompts: list[str] = []
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return self.completion


def _hit(
    chunk_id: str,
    text: str,
    *,
    doc_name: str = "Human Rights Policy",
    section: str,
    version: str,
    status: str,
    score: float = 1.0,
) -> RerankedHit:
    return RerankedHit(
        chunk_id=chunk_id,
        text=text,
        score=score,
        metadata={
            "doc_name": doc_name,
            "section": section,
            "version": version,
            "status": status,
        },
        rrf_score=0.02,
        dense_rank=1,
        bm25_rank=1,
        dense_distance=0.1,
        bm25_score=1.0,
    )


def _pto_window() -> list[RerankedHit]:
    return [
        _hit(
            "hr-v2",
            _CURRENT_PTO,
            section="5. Fair Wages and Remuneration",
            version=config.CURRENT_POLICY_VERSION,
            status="current",
            score=2.0,
        ),
        _hit(
            "hr-v1",
            _LEGACY_PTO,
            section="3. Fair Wages and Remuneration",
            version=config.LEGACY_POLICY_VERSION,
            status="legacy",
            score=1.5,
        ),
    ]


def _json_answer(*, grounded: bool, answer: str) -> str:
    return json.dumps({"grounded": grounded, "answer": answer})


def test_pto_conflict_answers_from_current_and_marks_legacy() -> None:
    leaked = f"Employees receive {config.LEGACY_PTO_DAYS} days of Privilege Leave."
    fake = _FakeGenerator(_json_answer(grounded=True, answer=leaked))
    answer = generate_answer(_PTO_QUESTION, _pto_window(), fake)
    rendered = render_answer(answer)

    assert fake.calls == 1
    assert answer.abstained is False
    assert answer.has_legacy_conflict is True
    assert answer.text == _CURRENT_PTO_SENTENCE
    assert f"{config.LEGACY_PTO_DAYS} days" not in answer.text
    assert f"{config.LEGACY_PTO_DAYS} days" not in rendered

    current = next(item for item in answer.citations if item.version == "2.0")
    legacy = next(item for item in answer.citations if item.version == "1.0")
    assert current.section == "5. Fair Wages and Remuneration"
    assert current.legacy_conflict is False
    assert legacy.section == "3. Fair Wages and Remuneration"
    assert legacy.legacy_conflict is True
    assert rendered == (
        f"{_CURRENT_PTO_SENTENCE}\n"
        "Sources:\n"
        "- Human Rights Policy, 5. Fair Wages and Remuneration, v2.0\n"
        "- Human Rights Policy, 3. Fair Wages and Remuneration, v1.0 (legacy conflict)\n"
    )
    prompt = fake.prompts[0]
    assert _CURRENT_PTO in prompt
    assert _LEGACY_PTO in prompt
    assert "status=current" in prompt
    assert "status=legacy" in prompt
    assert _DECOY not in prompt


def test_posh_email_is_cited() -> None:
    fake = _FakeGenerator(_json_answer(grounded=True, answer=_POSH_TEXT))
    hits = [
        _hit(
            "posh-1",
            _POSH_TEXT,
            doc_name="POSH Policy",
            section="4. Complaint Procedure",
            version="1.0",
            status="current",
        )
    ]
    answer = generate_answer("Where do I send a POSH complaint?", hits, fake)
    rendered = render_answer(answer)

    assert answer.abstained is False
    assert answer.has_legacy_conflict is False
    assert "shrc@coforge.com" in answer.text
    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert citation.doc_name == "POSH Policy"
    assert citation.section == "4. Complaint Procedure"
    assert citation.version == "1.0"
    assert citation.legacy_conflict is False
    assert "- POSH Policy, 4. Complaint Procedure, v1.0" in rendered
    assert "(legacy conflict)" not in rendered


def test_unsupported_chunks_abstain_without_an_invented_fact() -> None:
    invented = "The CEO personal mobile number is 5551234567."
    fake = _FakeGenerator(_json_answer(grounded=True, answer=invented))
    hits = [
        _hit(
            "board-1",
            "Board sitting fees are paid quarterly.",
            doc_name="Board Diversity Policy",
            section="2. Sitting Fees",
            version="1.0",
            status="current",
        )
    ]
    answer = generate_answer("What is the personal mobile number of the Coforge CEO?", hits, fake)

    assert answer.abstained is True
    assert answer.text == ABSTAIN_TEXT
    assert "5551234567" not in answer.text
    assert invented not in answer.text


def test_legacy_only_window_abstains_and_skips_the_model() -> None:
    fake = _FakeGenerator(_json_answer(grounded=True, answer="should not run"))
    hits = [
        _hit(
            "hr-v1-only",
            _LEGACY_PTO,
            section="3. Fair Wages and Remuneration",
            version=config.LEGACY_POLICY_VERSION,
            status="legacy",
        )
    ]
    answer = generate_answer(_PTO_QUESTION, hits, fake)

    assert fake.calls == 0
    assert answer.abstained is True
    assert answer.has_legacy_conflict is True
    assert answer.text == ABSTAIN_TEXT
    assert f"{config.LEGACY_PTO_DAYS} days" not in answer.text
    assert answer.citations[0].legacy_conflict is True
    assert answer.citations[0].version == config.LEGACY_POLICY_VERSION


def test_empty_question_raises() -> None:
    fake = _FakeGenerator(_json_answer(grounded=True, answer="unused"))
    with pytest.raises(GenerationError, match="empty"):
        generate_answer("   ", _pto_window(), fake)
    assert fake.calls == 0


def test_missing_env_raises_generation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.LLM_API_KEY_ENV, raising=False)
    monkeypatch.delenv(config.LLM_MODEL_ENV, raising=False)
    with pytest.raises(GenerationError, match=config.LLM_API_KEY_ENV):
        generator_from_env()

    monkeypatch.setenv(config.LLM_API_KEY_ENV, "sk-test")
    with pytest.raises(GenerationError, match=config.LLM_MODEL_ENV):
        generator_from_env()


def test_bad_model_json_raises() -> None:
    fake = _FakeGenerator("not-json")
    with pytest.raises(GenerationError, match="unparseable JSON"):
        generate_answer(_PTO_QUESTION, _pto_window(), fake)


@pytest.mark.integration
def test_live_openai_pto_conflict_blocks_legacy_days() -> None:
    if not os.environ.get(config.LLM_API_KEY_ENV, "").strip():
        pytest.skip("OPENAI_API_KEY is unset")
    if not os.environ.get(config.LLM_MODEL_ENV, "").strip():
        pytest.skip(f"{config.LLM_MODEL_ENV} is unset")

    answer = generate_answer(_PTO_QUESTION, _pto_window(), generator_from_env())
    rendered = render_answer(answer)
    assert f"{config.LEGACY_PTO_DAYS} days" not in answer.text
    assert f"{config.LEGACY_PTO_DAYS} days" not in rendered
    assert any(
        item.version == config.CURRENT_POLICY_VERSION and not item.legacy_conflict
        for item in answer.citations
    )
    assert any(
        item.version == config.LEGACY_POLICY_VERSION and item.legacy_conflict
        for item in answer.citations
    )
    assert answer.has_legacy_conflict is True
