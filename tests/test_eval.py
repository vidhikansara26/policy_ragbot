"""Recall@K and answer accuracy for the reranked policy retriever.

Gold spans are checked against ``data/raw``. Store tests use a fake pair
scorer and ``tmp_path`` so they do not download models or share the repo
``chroma/`` directory. The live MiniLM + cross-encoder run is marked
``integration``. Recall and accuracy are asserted in separate tests.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from rag_lab import config
from rag_lab.chunking import chunk_corpus
from rag_lab.corpus import load_corpus
from rag_lab.eval import (
    AnswerGold,
    EvalCase,
    EvalReport,
    GoldPassage,
    RetrievalLabel,
    answer_cases,
    bind_answer_gold,
    bind_gold,
    evaluate,
    evaluate_answers,
    evaluate_retrieval,
    evaluation_cases,
)
from rag_lab.exceptions import EvalError
from rag_lab.generate import generator_from_env
from rag_lab.hybrid import HybridRetriever
from rag_lab.index import PolicyIndex
from rag_lab.rerank import RerankedHit, RerankingRetriever


class _LengthEmbedder:
    """Stable 3-d vectors so a full-corpus store test does not load MiniLM."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.5]


class _RecordingScorer:
    """Fake pair scorer. Records every batch and never loads a model."""

    def __init__(self, score_for: Callable[[str, str], float]) -> None:
        self.batches: list[tuple[str, tuple[str, ...]]] = []
        self._score_for = score_for

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        self.batches.append((query, tuple(texts)))
        return [float(self._score_for(query, text)) for text in texts]


def _hit(chunk_id: str, text: str, metadata: dict[str, str], score: float) -> RerankedHit:
    return RerankedHit(
        chunk_id=chunk_id,
        text=text,
        score=score,
        metadata=metadata,
        rrf_score=0.02,
        dense_rank=1,
        bm25_rank=1,
        dense_distance=0.1,
        bm25_score=1.0,
    )


def _meta(doc_name: str, section: str, version: str, status: str) -> dict[str, str]:
    return {
        "doc_name": doc_name,
        "section": section,
        "version": version,
        "status": status,
    }


def _offline_report(
    directory: Path,
    score_for: Callable[[str, str], float],
) -> tuple[EvalReport, _RecordingScorer]:
    """Index the corpus in ``directory`` and evaluate the production query set."""
    documents = load_corpus()
    gold = bind_gold(evaluation_cases(), documents)
    scorer = _RecordingScorer(score_for)
    index = PolicyIndex(directory / "chroma", embedder=_LengthEmbedder())
    try:
        retriever = RerankingRetriever(HybridRetriever(index), scorer=scorer)
        retriever.upsert(chunk_corpus(documents))
        report = evaluate(retriever, gold)
    finally:
        index.close()
    return report, scorer


def _boost_gold_span(spans: dict[str, str]) -> Callable[[str, str], float]:
    def score_for(query: str, text: str) -> float:
        return 10.0 if spans[query] in text else 0.0

    return score_for


def test_eval_k_is_the_window_search_returns() -> None:
    """Recall uses the 5 returned hits, not the 20 that were only rescored."""
    assert config.EVAL_K == config.RETRIEVE_K
    assert config.EVAL_K == 5
    assert config.RERANK_CANDIDATE_K == 20
    assert config.EVAL_K < config.RERANK_CANDIDATE_K


def test_query_set_has_the_conflict_and_the_clean_facts() -> None:
    """POSH, whistleblower, EHS, and both Human Rights PTO wordings."""
    gold = bind_gold(evaluation_cases(), load_corpus())
    assert len(gold) >= 8
    files = {passage.source_file for passage in gold}
    assert "human_rights_policy_v1.md" in files
    assert "human_rights_policy_v2.md" in files
    assert "posh_policy.md" in files
    assert "whistleblower_policy.md" in files
    assert "ehs_policy.md" in files
    fixtures = [passage for passage in gold if passage.is_data_quality_fixture]
    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert fixture.query_id == "pto_privilege_leave"
    assert fixture.status == "legacy"
    assert fixture.version == config.LEGACY_POLICY_VERSION
    assert fixture.doc_name == config.PLANTED_DOC_NAME
    assert fixture.section == "3. Fair Wages and Remuneration"
    assert f"{config.LEGACY_PTO_DAYS} days of Privilege Leave (PTO)" == fixture.span
    current = next(
        passage for passage in gold if passage.query_id == "pto_current_no_numeric_entitlement"
    )
    assert current.status == "current"
    assert current.version == config.CURRENT_POLICY_VERSION
    assert current.section == "5. Fair Wages and Remuneration"
    assert current.is_data_quality_fixture is False


@pytest.mark.parametrize("case", evaluation_cases(), ids=lambda case: case.query_id)
def test_gold_span_is_verbatim_in_one_raw_file(case: EvalCase) -> None:
    """The label is the file text and the heading, not a hand-written fact."""
    passage = bind_gold([case], load_corpus())[0]
    raw = (config.RAW_DATA_DIR / passage.source_file).read_text(encoding="utf-8")
    assert case.span in raw
    assert f"## {passage.section}" in raw
    assert f'version: "{passage.version}"' in raw or f"version: {passage.version}" in raw
    assert f'status: "{passage.status}"' in raw or f"status: {passage.status}" in raw
    assert passage.doc_name
    assert passage.section
    assert passage.version


def test_blank_span_and_unknown_span_are_rejected() -> None:
    with pytest.raises(EvalError, match="empty"):
        EvalCase("days", "How many Privilege Leave days?", "  ")
    missing = EvalCase(
        "guessed",
        "How many Privilege Leave days are published?",
        "99 days of Privilege Leave",
    )
    with pytest.raises(EvalError, match="data/raw"):
        bind_gold([missing], load_corpus())


def test_duplicate_query_id_is_rejected() -> None:
    case = evaluation_cases()[0]
    with pytest.raises(EvalError, match="Duplicate"):
        bind_gold([case, case], load_corpus())


def test_recall_can_pass_when_answer_accuracy_fails() -> None:
    """Rank 3-style miss: the gold passage is in the window and is not the answer."""
    documents = load_corpus()
    cases = [
        case
        for case in evaluation_cases()
        if case.query_id in {"pto_privilege_leave", "ehs_net_zero"}
    ]
    gold = bind_gold(cases, documents)
    by_id = {passage.query_id: passage for passage in gold}
    pto = by_id["pto_privilege_leave"]
    ehs = by_id["ehs_net_zero"]
    pto_hit = _hit(
        "v1-wages",
        f"Employees receive {pto.span} each year.",
        _meta(pto.doc_name, pto.section, pto.version, pto.status),
        1.0,
    )
    distractor = _hit(
        "supplier-leave",
        "Suppliers offer adequate rest periods and parental leave.",
        _meta(
            "Supplier Code of Conduct",
            "Labor Management and Human Rights",
            "2025",
            "current",
        ),
        9.0,
    )
    ehs_hit = _hit(
        "ehs-objectives",
        f"The objectives include {ehs.span} for emissions.",
        _meta(ehs.doc_name, ehs.section, ehs.version, ehs.status),
        4.0,
    )

    class _Split:
        def search(self, query: str, *, k: int) -> list[RerankedHit]:
            assert k == config.EVAL_K
            if query == pto.question:
                return [distractor, pto_hit]
            assert query == ehs.question
            return [ehs_hit]

    report = evaluate(_Split(), gold)
    pto_result = report.by_id("pto_privilege_leave")
    ehs_result = report.by_id("ehs_net_zero")

    assert pto_result.recall_at_k == 1.0
    assert pto_result.answer_correct is False
    assert pto_result.retrieval_label is RetrievalLabel.DATA_QUALITY_FIXTURE
    assert pto_result.answer.doc_name == "Supplier Code of Conduct"
    assert pto_result.answer.section
    assert pto_result.answer.version
    assert "not a failed retrieval" in pto_result.note
    assert config.LEGACY_POLICY_VERSION in pto_result.versions_for(config.PLANTED_DOC_NAME)
    assert any(row.status == "legacy" for row in pto_result.retrieved)
    assert any(row.status == "current" for row in pto_result.retrieved)

    assert ehs_result.recall_at_k == 1.0
    assert ehs_result.answer_correct is True
    assert ehs_result.retrieval_label is RetrievalLabel.HIT
    assert ehs_result.answer.doc_name
    assert ehs_result.answer.section
    assert ehs_result.answer.version

    assert report.mean_recall_at_k == pytest.approx(1.0)
    assert report.answer_accuracy == pytest.approx(0.5)


def test_fixture_is_a_miss_only_when_v1_is_absent() -> None:
    documents = load_corpus()
    gold = bind_gold(
        [case for case in evaluation_cases() if case.is_data_quality_fixture],
        documents,
    )
    passage = gold[0]
    distractor = _hit(
        "supplier-leave",
        "Suppliers offer adequate rest periods and parental leave.",
        _meta(
            "Supplier Code of Conduct",
            "Labor Management and Human Rights",
            "2025",
            "current",
        ),
        3.0,
    )

    class _OnlyDistractor:
        def search(self, query: str, *, k: int) -> list[RerankedHit]:
            assert query == passage.question
            assert k == config.EVAL_K
            return [distractor]

    result = evaluate(_OnlyDistractor(), gold).by_id(passage.query_id)
    assert result.recall_at_k == 0.0
    assert result.answer_correct is False
    assert result.retrieval_label is RetrievalLabel.MISS
    assert "not a failed retrieval" not in result.note
    assert result.answer.doc_name
    assert result.answer.section
    assert result.answer.version


def test_answer_missing_citation_or_an_empty_window_fails() -> None:
    gold = bind_gold(
        [case for case in evaluation_cases() if case.is_data_quality_fixture],
        load_corpus(),
    )
    passage = gold[0]
    blank = _hit(
        "bare",
        passage.span,
        _meta("", passage.section, passage.version, passage.status),
        1.0,
    )

    class _BlankName:
        def search(self, query: str, *, k: int) -> list[RerankedHit]:
            del query, k
            return [blank]

    class _Empty:
        def search(self, query: str, *, k: int) -> list[RerankedHit]:
            del query, k
            return []

    class _Unused:
        def search(self, query: str, *, k: int) -> list[RerankedHit]:
            del query, k
            raise AssertionError("k is validated before search")

    with pytest.raises(EvalError, match="doc_name"):
        evaluate(_BlankName(), gold)
    with pytest.raises(EvalError, match="no passages"):
        evaluate(_Empty(), gold)
    with pytest.raises(EvalError, match="Invalid k"):
        evaluate(_Unused(), gold, k=0)


def test_recall_at_k(tmp_path: Path) -> None:
    """Every gold passage is inside the returned window. Legacy v1 counts."""
    documents = load_corpus()
    gold = bind_gold(evaluation_cases(), documents)
    spans = {passage.question: passage.span for passage in gold}
    report, scorer = _offline_report(tmp_path, _boost_gold_span(spans))

    assert report.k == config.RETRIEVE_K
    assert len(scorer.batches) == len(gold)
    assert {len(texts) for _, texts in scorer.batches} == {config.RERANK_CANDIDATE_K}
    missed = [result.query_id for result in report.results if result.recall_at_k != 1.0]
    assert missed == []
    assert all(len(result.retrieved) == config.RETRIEVE_K for result in report.results)
    assert report.mean_recall_at_k == pytest.approx(1.0)
    fixture = report.by_id("pto_privilege_leave")
    assert fixture.retrieval_label is RetrievalLabel.DATA_QUALITY_FIXTURE
    assert any(row.status == "legacy" and row.contains_gold_span for row in fixture.retrieved)


def test_answer_accuracy(tmp_path: Path) -> None:
    """Rank 1 cites the gold passage. Citation is doc name, section, and version."""
    documents = load_corpus()
    gold = bind_gold(evaluation_cases(), documents)
    spans = {passage.question: passage.span for passage in gold}
    report, _scorer = _offline_report(tmp_path, _boost_gold_span(spans))

    wrong = [result.query_id for result in report.results if not result.answer_correct]
    assert wrong == []
    assert report.answer_accuracy == pytest.approx(1.0)
    for result in report.results:
        assert result.answer.doc_name == result.gold.doc_name
        assert result.answer.section == result.gold.section
        assert result.answer.version == result.gold.version
        assert result.gold.span in result.answer.text
        assert result.answer.doc_name
        assert result.answer.section
        assert result.answer.version


def test_pto_legacy_hit_is_recorded_as_the_fixture(tmp_path: Path) -> None:
    """v1 in the window is the planted clause, not a failed retrieval."""
    documents = load_corpus()
    gold = bind_gold(evaluation_cases(), documents)
    spans = {passage.question: passage.span for passage in gold}
    report, _scorer = _offline_report(tmp_path, _boost_gold_span(spans))
    pto = report.by_id("pto_privilege_leave")
    current = report.by_id("pto_current_no_numeric_entitlement")

    assert pto.recall_at_k == 1.0
    assert pto.answer_correct is True
    assert pto.retrieval_label is RetrievalLabel.DATA_QUALITY_FIXTURE
    assert pto.retrieval_label is not RetrievalLabel.MISS
    assert pto.answer.doc_name == config.PLANTED_DOC_NAME
    assert pto.answer.section == "3. Fair Wages and Remuneration"
    assert pto.answer.version == config.LEGACY_POLICY_VERSION
    assert pto.answer.status == "legacy"
    assert f"{config.LEGACY_PTO_DAYS} days of Privilege Leave (PTO)" in pto.answer.text
    assert "version 1.0" in pto.note
    assert "status=legacy" in pto.note
    assert "not a failed retrieval" in pto.note
    assert config.LEGACY_POLICY_VERSION in pto.versions_for(config.PLANTED_DOC_NAME)

    assert current.retrieval_label is RetrievalLabel.HIT
    assert current.answer.version == config.CURRENT_POLICY_VERSION
    assert current.answer.status == "current"
    assert current.answer.section == "5. Fair Wages and Remuneration"
    assert "does not set a numeric Privilege Leave (PTO) entitlement" in current.answer.text


def test_recall_does_not_count_the_rescored_pool(tmp_path: Path) -> None:
    """A v1 clause scored in the pool of 20 and cut from the returned 5 is a miss."""
    case = next(case for case in evaluation_cases() if case.is_data_quality_fixture)
    gold = bind_gold([case], load_corpus())
    span = gold[0].span
    scorer = _RecordingScorer(lambda _query, text: -5.0 if span in text else 1.0)
    index = PolicyIndex(tmp_path / "chroma", embedder=_LengthEmbedder())
    try:
        retriever = RerankingRetriever(HybridRetriever(index), scorer=scorer)
        retriever.upsert(chunk_corpus(load_corpus()))
        report = evaluate(retriever, gold)
    finally:
        index.close()

    assert len(scorer.batches) == 1
    scored = scorer.batches[0][1]
    assert len(scored) == config.RERANK_CANDIDATE_K
    assert any(span in text for text in scored)
    result = report.by_id(case.query_id)
    assert result.k == config.RETRIEVE_K
    assert len(result.retrieved) == config.RETRIEVE_K
    assert result.recall_at_k == 0.0
    assert result.answer_correct is False
    assert result.retrieval_label is RetrievalLabel.MISS
    assert all(not row.contains_gold_span for row in result.retrieved)
    assert "not a failed retrieval" not in result.note


@pytest.mark.integration
def test_live_rerank_recall_and_answer_accuracy(tmp_path: Path) -> None:
    """MiniLM + BM25 + the cross-encoder. v1 PTO stays a fixture hit."""
    documents = load_corpus()
    gold = bind_gold(evaluation_cases(), documents)
    index = PolicyIndex(tmp_path / "chroma")
    try:
        retriever = RerankingRetriever(HybridRetriever(index))
        retriever.upsert(chunk_corpus(documents))
        report = evaluate(retriever, gold, k=config.EVAL_K)
    finally:
        index.close()

    assert report.k == config.RETRIEVE_K
    recall_misses = [result.query_id for result in report.results if result.recall_at_k != 1.0]
    accuracy_misses = [result.query_id for result in report.results if not result.answer_correct]
    assert recall_misses == []
    assert accuracy_misses == []
    assert report.mean_recall_at_k == pytest.approx(1.0)
    assert report.answer_accuracy == pytest.approx(1.0)

    pto = report.by_id("pto_privilege_leave")
    assert pto.retrieval_label is RetrievalLabel.DATA_QUALITY_FIXTURE
    assert pto.answer.doc_name == config.PLANTED_DOC_NAME
    assert pto.answer.section == "3. Fair Wages and Remuneration"
    assert pto.answer.version == config.LEGACY_POLICY_VERSION
    assert pto.answer.status == "legacy"
    assert f"{config.LEGACY_PTO_DAYS} days of Privilege Leave (PTO)" in pto.answer.text
    assert "not a failed retrieval" in pto.note
    assert pto.answer.score == pytest.approx(6.41, abs=0.05)
    versions = pto.versions_for(config.PLANTED_DOC_NAME)
    assert config.LEGACY_POLICY_VERSION in versions
    assert config.CURRENT_POLICY_VERSION in versions
    for result in report.results:
        assert result.answer.doc_name
        assert result.answer.section
        assert result.answer.version


class _ScriptedGenerator:
    """Grounded answer facts, with optional overrides keyed by a prompt fragment."""

    def __init__(self, overrides: dict[str, str] | None = None) -> None:
        self._overrides = overrides or {}
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        for fragment, completion in self._overrides.items():
            if fragment in prompt:
                return completion
        for case in answer_cases():
            if f"Question: {case.question}" in prompt:
                if case.must_abstain:
                    return _json_answer(grounded=False, answer="")
                return _json_answer(grounded=True, answer=case.span.replace("**", ""))
        raise AssertionError("prompt did not match a known question")


class _StaticSearcher:
    """Returns one fixed window for every question."""

    def __init__(self, hits: Sequence[RerankedHit]) -> None:
        self._hits = list(hits)

    def search(self, query: str, *, k: int) -> list[RerankedHit]:
        del query
        return self._hits[:k]


def _json_answer(*, grounded: bool, answer: str) -> str:
    return json.dumps({"grounded": grounded, "answer": answer})


def _boost_span_and_fact(
    passages: Sequence[GoldPassage],
    answers: Sequence[AnswerGold],
    *,
    abstain_score: float = -11.0,
) -> Callable[[str, str], float]:
    """Rank the retrieval span first, then the generation fact.

    Abstain rows score below the relevance floor.
    """
    spans = {passage.question: passage.span for passage in passages}
    facts = {row.question: row.fact for row in answers if not row.must_abstain}
    abstain = {row.question for row in answers if row.must_abstain}

    def score_for(query: str, text: str) -> float:
        if query in abstain:
            return abstain_score
        if query in spans and spans[query] in text:
            return 10.0
        fact = facts.get(query, "")
        if fact and fact in text:
            return 9.0
        return 0.0

    return score_for


def test_supported_set_keeps_ten_questions_and_separate_generation_gold() -> None:
    cases = evaluation_cases()
    assert len(cases) >= 10
    fixture = next(case for case in cases if case.is_data_quality_fixture)
    assert fixture.span == f"{config.LEGACY_PTO_DAYS} days of Privilege Leave (PTO)"
    generated = answer_cases()
    supported = [case for case in generated if not case.must_abstain]
    assert len(supported) >= 10
    assert [case.query_id for case in generated if case.must_abstain] == [
        "out_of_domain_wifi",
        "unsupported_ceo_phone",
    ]
    pto = next(case for case in generated if case.query_id == "pto_privilege_leave")
    assert pto.question == fixture.question
    assert pto.legacy_conflict is True
    assert pto.span == "does not set a numeric Privilege Leave (PTO) entitlement"
    assert pto.span != fixture.span
    documents = load_corpus()
    retrieval = bind_gold([fixture], documents)[0]
    answer = bind_answer_gold([pto], documents)[0]
    assert retrieval.status == "legacy"
    assert retrieval.version == config.LEGACY_POLICY_VERSION
    assert answer.status == "current"
    assert answer.version == config.CURRENT_POLICY_VERSION
    assert answer.fact == pto.span


def test_generated_scores_are_separate_and_perfect_with_a_scripted_generator(
    tmp_path: Path,
) -> None:
    """Recall@5 stays on the retrieval report. The other four checks score the published answer."""
    documents = load_corpus()
    retrieval_gold = bind_gold(evaluation_cases(), documents)
    answer_gold = bind_answer_gold(answer_cases(), documents)
    scorer = _RecordingScorer(_boost_span_and_fact(retrieval_gold, answer_gold))
    index = PolicyIndex(tmp_path / "chroma", embedder=_LengthEmbedder())
    try:
        retriever = RerankingRetriever(HybridRetriever(index), scorer=scorer)
        retriever.upsert(chunk_corpus(documents))
        retrieval = evaluate_retrieval(retriever, retrieval_gold)
        report = evaluate_answers(retriever, answer_gold, _ScriptedGenerator())
    finally:
        index.close()

    assert report.k == config.EVAL_K
    assert report.generated_key_fact == pytest.approx(1.0)
    assert report.citation_complete == pytest.approx(1.0)
    assert report.groundedness == pytest.approx(1.0)
    assert report.conflict_handling == pytest.approx(1.0)
    assert len(report.results) == len(answer_gold)

    fixture = retrieval.by_id("pto_privilege_leave")
    assert fixture.recall_at_k == 1.0
    assert fixture.answer_correct is True
    assert fixture.retrieval_label is RetrievalLabel.DATA_QUALITY_FIXTURE
    assert f"{config.LEGACY_PTO_DAYS} days of Privilege Leave (PTO)" in fixture.answer.text

    generated = report.by_id("pto_privilege_leave")
    assert generated.key_fact_correct is True
    assert f"{config.LEGACY_PTO_DAYS} days" not in generated.prose
    assert "(legacy conflict)" in generated.published
    assert "Fair Wages and Remuneration, v2.0" in generated.published
    assert "Fair Wages and Remuneration, v1.0 (legacy conflict)" in generated.published

    posh = report.by_id("posh_shrc_email")
    assert "(legacy conflict)" not in posh.published
    assert "shrc@coforge.com" in posh.prose

    wifi = report.by_id("out_of_domain_wifi")
    assert wifi.prose == "I cannot answer from the retrieved policies."
    assert wifi.key_fact_correct is True
    assert wifi.grounded is True
    assert "- (none)" in wifi.published

    refused = report.by_id("unsupported_ceo_phone")
    assert refused.key_fact_correct is True
    assert refused.grounded is True
    assert refused.prose == "I cannot answer from the retrieved policies."


def test_planted_day_count_fails_key_fact_and_conflict_handling() -> None:
    """A published 15-day claim fails generation and leaves the v1 retrieval fixture intact."""
    documents = load_corpus()
    fixture = next(case for case in evaluation_cases() if case.is_data_quality_fixture)
    retrieval_gold = bind_gold([fixture], documents)
    passage = retrieval_gold[0]
    answer_row = bind_answer_gold(
        [case for case in answer_cases() if case.query_id == fixture.query_id],
        documents,
    )[0]
    current = (
        f"{answer_row.fact} {config.LEGACY_PTO_DAYS} days appears in this "
        "current text only so the leak guard leaves the model sentence in place."
    )
    legacy = f"Full-time employees receive {passage.span} per calendar year."
    hits = [
        _hit(
            "legacy",
            legacy,
            _meta(passage.doc_name, passage.section, passage.version, passage.status),
            3.0,
        ),
        _hit(
            "current",
            current,
            _meta(answer_row.doc_name, answer_row.section, answer_row.version, answer_row.status),
            2.0,
        ),
    ]
    answer = (
        f"Employees receive {config.LEGACY_PTO_DAYS} days of Privilege Leave. {answer_row.fact}"
    )
    searcher = _StaticSearcher(hits)
    generator = _ScriptedGenerator({fixture.question: _json_answer(grounded=True, answer=answer)})
    report = evaluate_answers(searcher, [answer_row], generator)
    row = report.by_id(fixture.query_id)
    assert f"{config.LEGACY_PTO_DAYS} days" in row.prose
    assert row.key_fact_correct is False
    assert row.conflict_handled is False

    retrieval = evaluate_retrieval(searcher, retrieval_gold)
    retrieved = retrieval.by_id(fixture.query_id)
    assert retrieved.recall_at_k == 1.0
    assert retrieved.answer_correct is True
    assert retrieved.retrieval_label is RetrievalLabel.DATA_QUALITY_FIXTURE
    assert retrieval.answer_accuracy == pytest.approx(1.0)


def test_wifi_abstains_without_calling_the_model() -> None:
    """A negative-score distractor never reaches the scripted password."""
    documents = load_corpus()
    gold = bind_answer_gold(
        [case for case in answer_cases() if case.query_id == "out_of_domain_wifi"],
        documents,
    )
    wifi = _hit(
        "privacy-wifi",
        "Data privacy covers employee personal information, not office network credentials.",
        _meta("Data Privacy Policy", "1. Scope", "1.0", "current"),
        -11.2,
    )
    generator = _ScriptedGenerator(
        {"Wi-Fi": _json_answer(grounded=True, answer="The password is posted in the lobby.")}
    )
    report = evaluate_answers(_StaticSearcher([wifi]), gold, generator)
    row = report.by_id("out_of_domain_wifi")
    assert generator.calls == 0
    assert row.prose == "I cannot answer from the retrieved policies."
    assert "password" not in row.prose
    assert row.key_fact_correct is True
    assert row.grounded is True
    assert row.conflict_handled is True
    assert row.citation_complete is True
    assert row.published.endswith("Sources:\n- (none)\n")


def test_refusing_a_supported_question_fails_key_fact_and_groundedness(
    tmp_path: Path,
) -> None:
    documents = load_corpus()
    retrieval_gold = bind_gold(evaluation_cases(), documents)
    answer_gold = bind_answer_gold(answer_cases(), documents)
    generator = _ScriptedGenerator({"net zero": _json_answer(grounded=False, answer="")})
    index = PolicyIndex(tmp_path / "chroma", embedder=_LengthEmbedder())
    try:
        retriever = RerankingRetriever(
            HybridRetriever(index),
            scorer=_RecordingScorer(_boost_span_and_fact(retrieval_gold, answer_gold)),
        )
        retriever.upsert(chunk_corpus(documents))
        report = evaluate_answers(retriever, answer_gold, generator)
    finally:
        index.close()

    missed = report.by_id("ehs_net_zero")
    assert missed.key_fact_correct is False
    assert missed.grounded is False
    assert report.generated_key_fact < 1.0
    assert report.groundedness < 1.0


def test_invented_phone_number_is_not_published(tmp_path: Path) -> None:
    """A grounded=true invention is dropped. The refusal row stays the abstain sentence."""
    documents = load_corpus()
    retrieval_gold = bind_gold(evaluation_cases(), documents)
    answer_gold = bind_answer_gold(answer_cases(), documents)
    generator = _ScriptedGenerator(
        {
            "Coforge CEO": _json_answer(
                grounded=True,
                answer="The CEO personal mobile number is 5551234567.",
            )
        }
    )
    index = PolicyIndex(tmp_path / "chroma", embedder=_LengthEmbedder())
    try:
        retriever = RerankingRetriever(
            HybridRetriever(index),
            scorer=_RecordingScorer(
                _boost_span_and_fact(retrieval_gold, answer_gold, abstain_score=0.0)
            ),
        )
        retriever.upsert(chunk_corpus(documents))
        report = evaluate_answers(retriever, answer_gold, generator)
    finally:
        index.close()

    refused = report.by_id("unsupported_ceo_phone")
    assert generator.calls >= 1
    assert refused.prose == "I cannot answer from the retrieved policies."
    assert "5551234567" not in refused.prose
    assert refused.grounded is True
    assert refused.key_fact_correct is True


@pytest.mark.integration
def test_live_generator_pto_conflict_row() -> None:
    """One live answer. Skip when OpenAI is not configured."""
    if not os.environ.get(config.LLM_API_KEY_ENV, "").strip():
        pytest.skip("OPENAI_API_KEY is unset")
    if not os.environ.get(config.LLM_MODEL_ENV, "").strip():
        pytest.skip(f"{config.LLM_MODEL_ENV} is unset")

    documents = load_corpus()
    fixture = next(case for case in evaluation_cases() if case.is_data_quality_fixture)
    passage = bind_gold([fixture], documents)[0]
    answer_row = bind_answer_gold(
        [case for case in answer_cases() if case.query_id == fixture.query_id],
        documents,
    )[0]
    current = (
        "This current policy does not set a numeric Privilege Leave (PTO) entitlement. "
        "Leave follows local employment law."
    )
    legacy = f"Full-time employees are entitled to {passage.span} per calendar year."
    hits = [
        _hit(
            "hr-v2",
            current,
            _meta(answer_row.doc_name, answer_row.section, answer_row.version, answer_row.status),
            2.0,
        ),
        _hit(
            "hr-v1",
            legacy,
            _meta(passage.doc_name, passage.section, passage.version, passage.status),
            1.0,
        ),
    ]
    report = evaluate_answers(_StaticSearcher(hits), [answer_row], generator_from_env())
    row = report.by_id(fixture.query_id)
    assert f"{config.LEGACY_PTO_DAYS} days" not in row.published
    assert answer_row.fact in row.prose or "does not specify a numeric" in row.prose
    assert "(legacy conflict)" in row.published
