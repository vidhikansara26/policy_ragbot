"""The committed eval snapshot must stay in the shape CI and the PDF expect.

This does not re-run MiniLM or a language model. The file is the scoreboard
from ``python -m rag_lab eval`` against ``qwen3:8b``. Regenerating it is
``python -m rag_lab eval --output eval/record.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_lab.eval import answer_cases, evaluation_cases

_RECORD = Path(__file__).resolve().parents[1] / "eval" / "record.json"
_CITATION_FIELDS = ("doc_name", "section", "version", "status")


def test_committed_eval_record_has_perfect_recall_and_generated_scores() -> None:
    payload = json.loads(_RECORD.read_text(encoding="utf-8"))
    retrieval = payload["retrieval"]
    generated = payload["generated"]

    assert retrieval["k"] == 5
    assert retrieval["recall_at_k"] == 1.0
    assert retrieval["extractive_answer_accuracy"] == 1.0
    assert generated["generated_key_fact"] == 1.0
    assert generated["citation_complete"] == 1.0
    assert generated["groundedness"] == 1.0
    assert generated["conflict_handling"] == 1.0

    rows = retrieval["results"]
    expected_ids = [case.query_id for case in evaluation_cases()]
    assert [row["query_id"] for row in rows] == expected_ids
    questions = {case.query_id: case.question for case in evaluation_cases()}
    for row in rows:
        assert row["question"] == questions[row["query_id"]]
        assert row["recall_at_k"] == 1.0
        assert row["answer_correct"] is True
        cited = row["cited"]
        assert all(cited[field] for field in _CITATION_FIELDS)

    fixture = next(row for row in rows if row["query_id"] == "pto_privilege_leave")
    assert fixture["retrieval_label"] == "data_quality_fixture"
    assert fixture["cited"]["version"] == "1.0"
    assert fixture["cited"]["status"] == "legacy"
    assert "not a failed retrieval" in fixture["note"]

    generated_ids = {case.query_id for case in answer_cases()}
    assert "out_of_domain_wifi" in generated_ids
    assert payload["captured_from"] == "python -m rag_lab eval"
    assert payload["llm"]["model"] == "qwen3:8b"
