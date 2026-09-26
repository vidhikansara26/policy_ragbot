"""Generation-time relevance screen. Does not touch ingest."""

from __future__ import annotations

import json

from rag_lab import config
from rag_lab.generate import ABSTAIN_TEXT, generate_answer
from rag_lab.rerank import RerankedHit
from rag_lab.safety import screen_hits

_CURRENT_PTO = (
    "This current policy does not set a numeric Privilege Leave (PTO) entitlement. "
    "Leave, rest days, overtime, and other statutory entitlements follow the "
    "applicable local employment law and the employee's service conditions."
)
_LEGACY_PTO = (
    f"Full-time employees are entitled to {config.LEGACY_PTO_DAYS} days of "
    "Privilege Leave (PTO) per calendar year."
)
_SUPPLIER_LEAVE = (
    "Offer fair compensation and working conditions, including adequate rest "
    "periods and parental leave that align with prevailing standards, and "
    "implement a grievance redressal mechanism with non-retaliation protection."
)
_WIFI_QUESTION = "What is the Wi-Fi password?"


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
    doc_name: str,
    section: str,
    version: str,
    status: str,
    score: float,
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


def _headers(prompt: str) -> list[str]:
    return [line for line in prompt.splitlines() if line.startswith("[")]


def test_negative_score_window_abstains_without_the_model() -> None:
    wifi = _hit(
        "privacy-wifi",
        "Data privacy covers employee personal information, not office network credentials.",
        doc_name="Data Privacy Policy",
        section="1. Scope",
        version="1.0",
        status="current",
        score=-11.2,
    )
    screened = screen_hits(_WIFI_QUESTION, [wifi])
    assert screened.abstain is True
    assert screened.hits == ()
    assert screened.has_legacy_conflict is False

    fake = _FakeGenerator(
        json.dumps({"grounded": True, "answer": "The password is posted in the lobby."})
    )
    answer = generate_answer(_WIFI_QUESTION, [wifi], fake)
    assert fake.calls == 0
    assert answer.abstained is True
    assert answer.text == ABSTAIN_TEXT
    assert answer.citations == ()
    assert answer.has_legacy_conflict is False


def test_human_rights_conflict_prefers_current_over_higher_scored_legacy() -> None:
    legacy = _hit(
        "hr-v1",
        _LEGACY_PTO,
        doc_name=config.PLANTED_DOC_NAME,
        section="3. Fair Wages and Remuneration",
        version=config.LEGACY_POLICY_VERSION,
        status="legacy",
        score=6.0,
    )
    current = _hit(
        "hr-v2",
        _CURRENT_PTO,
        doc_name=config.PLANTED_DOC_NAME,
        section="5. Fair Wages and Remuneration",
        version=config.CURRENT_POLICY_VERSION,
        status="current",
        score=2.0,
    )
    screened = screen_hits("How many Privilege Leave / PTO days do I get?", [legacy, current])
    assert screened.abstain is False
    assert screened.has_legacy_conflict is True
    assert [hit.chunk_id for hit in screened.hits] == ["hr-v2", "hr-v1"]

    published = "This current policy does not set a numeric Privilege Leave (PTO) entitlement."
    fake = _FakeGenerator(json.dumps({"grounded": True, "answer": published}))
    answer = generate_answer(
        "How many Privilege Leave / PTO days do I get?",
        [legacy, current],
        fake,
    )

    assert fake.calls == 1
    assert answer.abstained is False
    assert answer.has_legacy_conflict is True
    headers = _headers(fake.prompts[0])
    assert headers[0].startswith("[1] ")
    assert "version=2.0" in headers[0]
    assert "status=current" in headers[0]
    assert "label=LEGACY" not in headers[0]
    assert headers[-1].startswith("[2] ")
    assert "version=1.0" in headers[-1]
    assert "status=legacy" in headers[-1]
    assert "label=LEGACY" in headers[-1]
    assert fake.prompts[0].index(_CURRENT_PTO) < fake.prompts[0].index(_LEGACY_PTO)


def test_legacy_survivor_re_admits_its_current_twin_from_below_the_floor() -> None:
    """The cross-encoder scores the retired clause above the policy that replaced it."""
    legacy = _hit(
        "hr-v1",
        _LEGACY_PTO,
        doc_name=config.PLANTED_DOC_NAME,
        section="3. Fair Wages and Remuneration",
        version=config.LEGACY_POLICY_VERSION,
        status="legacy",
        score=6.406,
    )
    current = _hit(
        "hr-v2",
        _CURRENT_PTO,
        doc_name=config.PLANTED_DOC_NAME,
        section="5. Fair Wages and Remuneration",
        version=config.CURRENT_POLICY_VERSION,
        status="current",
        score=-1.023,
    )
    assert current.score < config.MIN_RERANK_SCORE <= legacy.score

    question = "How many Privilege Leave / PTO days do I get?"
    screened = screen_hits(question, [legacy, current])
    assert screened.abstain is False
    assert screened.has_legacy_conflict is True
    assert [hit.chunk_id for hit in screened.hits] == ["hr-v2", "hr-v1"]

    published = "This current policy does not set a numeric Privilege Leave (PTO) entitlement."
    fake = _FakeGenerator(json.dumps({"grounded": True, "answer": published}))
    answer = generate_answer(question, [legacy, current], fake)

    assert fake.calls == 1
    assert answer.abstained is False
    assert answer.has_legacy_conflict is True
    assert f"{config.LEGACY_PTO_DAYS} days" not in answer.text
    legacy_citation = next(
        item for item in answer.citations if item.version == config.LEGACY_POLICY_VERSION
    )
    assert legacy_citation.legacy_conflict is True
    assert any(
        item.version == config.CURRENT_POLICY_VERSION and not item.legacy_conflict
        for item in answer.citations
    )


def test_companion_rule_does_not_re_admit_an_unrelated_document() -> None:
    """Only the legacy hit's own document is re-admitted from below the floor."""
    legacy = _hit(
        "hr-v1",
        _LEGACY_PTO,
        doc_name=config.PLANTED_DOC_NAME,
        section="3. Fair Wages and Remuneration",
        version=config.LEGACY_POLICY_VERSION,
        status="legacy",
        score=6.406,
    )
    supplier = _hit(
        "supplier-leave",
        _SUPPLIER_LEAVE,
        doc_name="Supplier Code of Conduct",
        section="Labor Management and Human Rights",
        version="2025",
        status="current",
        score=-4.2,
    )
    screened = screen_hits("How many Privilege Leave / PTO days do I get?", [legacy, supplier])
    assert [hit.chunk_id for hit in screened.hits] == ["hr-v1"]
    assert screened.has_legacy_conflict is False


def test_companion_rule_keeps_the_highest_scoring_current_twin() -> None:
    legacy = _hit(
        "hr-v1",
        _LEGACY_PTO,
        doc_name=config.PLANTED_DOC_NAME,
        section="3. Fair Wages and Remuneration",
        version=config.LEGACY_POLICY_VERSION,
        status="legacy",
        score=6.406,
    )
    weak = _hit(
        "hr-v2-preamble",
        "The Company respects internationally recognised human rights.",
        doc_name=config.PLANTED_DOC_NAME,
        section="1. Introduction",
        version=config.CURRENT_POLICY_VERSION,
        status="current",
        score=-6.5,
    )
    best = _hit(
        "hr-v2",
        _CURRENT_PTO,
        doc_name=config.PLANTED_DOC_NAME,
        section="5. Fair Wages and Remuneration",
        version=config.CURRENT_POLICY_VERSION,
        status="current",
        score=-1.023,
    )
    screened = screen_hits(
        "How many Privilege Leave / PTO days do I get?",
        [legacy, weak, best],
    )
    assert [hit.chunk_id for hit in screened.hits] == ["hr-v2", "hr-v1"]


def test_abstention_still_wins_when_nothing_clears_the_floor() -> None:
    """The companion rule cannot resurrect a window the floor rejected outright."""
    legacy = _hit(
        "hr-v1",
        _LEGACY_PTO,
        doc_name=config.PLANTED_DOC_NAME,
        section="3. Fair Wages and Remuneration",
        version=config.LEGACY_POLICY_VERSION,
        status="legacy",
        score=-0.5,
    )
    current = _hit(
        "hr-v2",
        _CURRENT_PTO,
        doc_name=config.PLANTED_DOC_NAME,
        section="5. Fair Wages and Remuneration",
        version=config.CURRENT_POLICY_VERSION,
        status="current",
        score=-1.023,
    )
    screened = screen_hits("How many Privilege Leave / PTO days do I get?", [legacy, current])
    assert screened.abstain is True
    assert screened.hits == ()


def test_short_pto_query_drops_supplier_leave_and_prefers_current() -> None:
    supplier = _hit(
        "supplier-leave",
        _SUPPLIER_LEAVE,
        doc_name="Supplier Code of Conduct",
        section="Labor Management and Human Rights",
        version="2025",
        status="current",
        score=4.51,
    )
    legacy = _hit(
        "hr-v1",
        _LEGACY_PTO,
        doc_name=config.PLANTED_DOC_NAME,
        section="3. Fair Wages and Remuneration",
        version=config.LEGACY_POLICY_VERSION,
        status="legacy",
        score=3.2,
    )
    current = _hit(
        "hr-v2",
        _CURRENT_PTO,
        doc_name=config.PLANTED_DOC_NAME,
        section="5. Fair Wages and Remuneration",
        version=config.CURRENT_POLICY_VERSION,
        status="current",
        score=2.4,
    )
    published = "This current policy does not set a numeric Privilege Leave (PTO) entitlement."
    fake = _FakeGenerator(json.dumps({"grounded": True, "answer": published}))
    answer = generate_answer("PTO", [supplier, legacy, current], fake)

    assert fake.calls == 1
    assert answer.abstained is False
    assert answer.has_legacy_conflict is True
    prompt = fake.prompts[0]
    assert _SUPPLIER_LEAVE not in prompt
    assert "Supplier Code of Conduct" not in prompt
    headers = _headers(prompt)
    assert "version=2.0" in headers[0]
    assert "status=current" in headers[0]
    assert config.PLANTED_DOC_NAME in headers[0]
    assert "version=1.0" in headers[-1]
    assert "label=LEGACY" in headers[-1]
    assert prompt.index(_CURRENT_PTO) < prompt.index(_LEGACY_PTO)
