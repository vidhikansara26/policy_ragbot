"""Command output for one query and for the eval scoreboard.

The store is a fake embedder, a fake pair scorer, and ``tmp_path``. These
tests do not download MiniLM or share the repo ``chroma/`` directory.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from rag_lab import config
from rag_lab.cli import Pipeline, build_pipeline, main, render_query
from rag_lab.corpus import load_corpus
from rag_lab.eval import bind_gold, evaluation_cases
from rag_lab.exceptions import EvalError
from rag_lab.rerank import RerankedHit


class _LengthEmbedder:
    """Stable 3-d vectors so the CLI test does not load MiniLM."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.5]


class _SpanScorer:
    """Boosts the gold span for the question under test."""

    def __init__(self, spans: dict[str, str]) -> None:
        self._spans = spans

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        span = self._spans.get(query, "")
        return [10.0 if span and span in text else 0.0 for text in texts]


def _opener(directory: Path) -> Callable[[], Pipeline]:
    documents = load_corpus()
    spans = {passage.question: passage.span for passage in bind_gold(evaluation_cases(), documents)}

    def open_pipeline() -> Pipeline:
        return build_pipeline(
            directory / "chroma",
            documents,
            embedder=_LengthEmbedder(),
            scorer=_SpanScorer(spans),
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


def test_eval_command_prints_recall_and_accuracy_separately(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["eval"], opener=_opener(tmp_path))
    captured = capsys.readouterr()
    assert code == 0
    output = captured.out
    assert "recall_at_k=1.000" in output
    assert "answer_accuracy=1.000" in output
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
