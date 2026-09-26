"""Command output for one query and for the eval scoreboard.

The store is a fake embedder, a fake pair scorer, and ``tmp_path``. These
tests do not download MiniLM or share the repo ``chroma/`` directory.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from test_hybrid import _chunk, _ScriptedEmbedder

from rag_lab import config
from rag_lab.cli import Pipeline, build_pipeline, main, render_query
from rag_lab.corpus import load_corpus
from rag_lab.eval import answer_cases, bind_answer_gold, bind_gold, evaluation_cases
from rag_lab.exceptions import EvalError
from rag_lab.hybrid import HybridRetriever
from rag_lab.index import PolicyIndex
from rag_lab.rerank import RerankedHit, RerankingRetriever


class _LengthEmbedder:
    """Stable 3-d vectors so the CLI test does not load MiniLM."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.5]


class _SpanScorer:
    """Boost the retrieval span, then the generation fact.

    Abstain questions score below the relevance floor.
    """

    def __init__(
        self,
        spans: dict[str, str],
        facts: dict[str, str] | None = None,
        abstain: set[str] | None = None,
    ) -> None:
        self._spans = spans
        self._facts = {} if facts is None else facts
        self._abstain = set() if abstain is None else abstain

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        if query in self._abstain:
            return [-11.0 for _text in texts]
        span = self._spans.get(query, "")
        fact = self._facts.get(query, "")
        scores: list[float] = []
        for text in texts:
            if span and span in text:
                scores.append(10.0)
            elif fact and fact in text:
                scores.append(9.0)
            else:
                scores.append(0.0)
        return scores


def _opener(directory: Path) -> Callable[[], Pipeline]:
    documents = load_corpus()
    spans = {passage.question: passage.span for passage in bind_gold(evaluation_cases(), documents)}
    answers = bind_answer_gold(answer_cases(), documents)
    facts = {row.question: row.fact for row in answers if not row.must_abstain}
    abstain = {row.question for row in answers if row.must_abstain}

    def open_pipeline() -> Pipeline:
        return build_pipeline(
            directory / "chroma",
            documents,
            embedder=_LengthEmbedder(),
            scorer=_SpanScorer(spans, facts, abstain),
        )

    return open_pipeline


def test_query_command_prints_chunks_and_cited_answer(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        ["query", "How many Privilege Leave / PTO days do I get?"],
        opener=_opener(tmp_path),
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    output = captured.out
    assert "Pipeline: chunk → MiniLM dense → Chroma → BM25 + reciprocal rank fusion" in output
    assert "Chunks" in output
    assert "Answer" in output
    assert "Human Rights Policy" in output
    assert "version 1.0" in output
    assert "section '3. Fair Wages and Remuneration'" in output
    assert "status=legacy" in output
    assert f"{config.LEGACY_PTO_DAYS} days of Privilege Leave (PTO)" in output
    assert f"Returned {config.RETRIEVE_K} chunks." in output


class _FactGenerator:
    """Returns the generation fact for each supported question and abstains on the rest."""

    def complete(self, prompt: str) -> str:
        for case in answer_cases():
            if f"Question: {case.question}" in prompt:
                if case.must_abstain:
                    return json.dumps({"grounded": False, "answer": ""})
                fact = case.span.replace("**", "")
                return json.dumps({"grounded": True, "answer": fact})
        raise AssertionError("eval prompt did not match a known question")


def test_eval_command_prints_recall_and_accuracy_separately(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["eval"], opener=_opener(tmp_path), generator_factory=_FactGenerator)
    captured = capsys.readouterr()
    assert code == 0
    output = captured.out
    assert "recall_at_k=1.000" in output
    assert "extractive_answer_accuracy=1.000" in output
    assert "generated_key_fact=1.000" in output
    assert "citation_complete=1.000" in output
    assert "groundedness=1.000" in output
    assert "conflict_handling=1.000" in output
    assert "recall_at_5=" not in output
    assert "key_fact_accuracy=" not in output
    assert "citation_completeness=" not in output
    assert f"k={config.EVAL_K}" in output
    assert "pto_privilege_leave  recall=1  accurate=1  data_quality_fixture" in output
    assert "not a failed retrieval" in output
    assert "Human Rights Policy  v1.0  legacy" in output
    assert "posh_shrc_email  recall=1  accurate=1  hit" in output
    assert "whistleblower_channel  recall=1  accurate=1  hit" in output
    assert "ehs_net_zero  recall=1  accurate=1  hit" in output


def test_query_without_tokens_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["query", "???"], opener=_opener(tmp_path))
    captured = capsys.readouterr()
    assert code == 1
    assert "tokens" in captured.err
    assert captured.out == ""


def test_blank_query_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["query", "   "], opener=_opener(tmp_path))
    captured = capsys.readouterr()
    assert code == 1
    assert "empty" in captured.err


def _scripted_compare_opener(directory: Path) -> Callable[[], Pipeline]:
    """Same three-chunk layout as the hybrid lexical-legacy fusion test."""
    legacy = _chunk(
        "legacy-pto",
        "Full-time employees are entitled to 15 days of Privilege Leave (PTO) per calendar year.",
        status="legacy",
        version="1.0",
        doc_name="Human Rights Policy",
    )
    dense_only = _chunk(
        "dense-board",
        "The board reviews director tenure and sitting fees each year.",
    )
    neither = _chunk(
        "annual-report",
        "Annual report publication calendar for the board.",
    )
    chunks = [legacy, dense_only, neither]
    vectors = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.6, 0.8, 0.0],
    ]
    query_vector = [1.0, 0.0, 0.0]

    def open_pipeline() -> Pipeline:
        index = PolicyIndex(
            directory / "chroma",
            embedder=_ScriptedEmbedder(vectors, query_vector),
        )
        hybrid = HybridRetriever(index)
        hybrid.upsert(chunks)
        return Pipeline(
            index=index,
            retriever=RerankingRetriever(hybrid, scorer=_SpanScorer({})),
            documents=(),
            chunk_count=len(chunks),
        )

    return open_pipeline


def test_compare_command_prints_dense_and_rrf_side_by_side(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Legacy PTO is dense rank 3, BM25 rank 1, and fused rank 1."""
    code = main(
        ["compare", "Privilege Leave PTO days", "--k", "3"],
        opener=_scripted_compare_opener(tmp_path),
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    output = captured.out
    assert "Query: Privilege Leave PTO days" in output
    assert "k=3" in output
    assert "dense_rank" in output
    assert "bm25_rank" in output
    row = next(line for line in output.splitlines() if line.startswith("1 "))
    dense_label = "Board Diversity Policy, Fair Wages and Remuneration, 2.0, current"
    hybrid_label = "Human Rights Policy, Fair Wages and Remuneration, 1.0, legacy"
    assert row.index(dense_label) < row.index(hybrid_label)
    tail = row.split(hybrid_label, maxsplit=1)[1].split()
    assert tail[0] == "3"
    assert tail[1] == "1"
    assert tail[2] == f"{1 / 63 + 1 / 61:.6f}"


def test_render_query_rejects_a_chunk_without_a_doc_name() -> None:
    hit = RerankedHit(
        chunk_id="bare",
        text="15 days of Privilege Leave (PTO)",
        score=1.0,
        metadata={"doc_name": "", "section": "Wages", "version": "1.0", "status": "legacy"},
        rrf_score=0.02,
        dense_rank=1,
        bm25_rank=1,
        dense_distance=0.1,
        bm25_score=1.0,
    )
    with pytest.raises(EvalError, match="doc_name"):
        render_query("Privilege Leave", [hit], documents=1, chunk_count=1)


@pytest.mark.integration
def test_compare_live_hr_mailbox_bm25_promotes_gold(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Live MiniLM compare for the first mailbox dense misses and BM25 ranks 1.

    ``shrc@coforge.com`` and ``whistleblower@coforge.com`` both sit in the
    dense top-5. ``All.HR@coforge.com`` does not: Human Rights Policy v2,
    section 13. Grievance Redressal, is dense rank 8, BM25 rank 1, and fused
    rank 2.
    """
    documents = load_corpus()

    def open_pipeline() -> Pipeline:
        return build_pipeline(tmp_path / "chroma", documents)

    code = main(["compare", "All.HR@coforge.com"], opener=open_pipeline)
    captured = capsys.readouterr()
    assert code == 0
    assert not captured.err.startswith("error:")
    output = captured.out
    assert "Query: All.HR@coforge.com" in output
    assert f"k={config.RETRIEVE_K}" in output
    gold = "Human Rights Policy, 13. Grievance Redressal, 2.0, current"
    rows = [line for line in output.splitlines() if line[:1].isdigit()]
    assert len(rows) == config.RETRIEVE_K
    matches = [line for line in rows if gold in line]
    assert len(matches) == 1
    row = matches[0]
    assert row.startswith("2")
    tail = row.split(gold, maxsplit=1)[1].split()
    assert tail == ["8", "1", f"{1 / (config.RRF_K + 8) + 1 / (config.RRF_K + 1):.6f}"]
