"""Demo UI over a fake embedder and a temporary Chroma directory.

These tests do not download MiniLM or the cross-encoder.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_cli import _FactGenerator, _opener

from rag_lab.cli import Pipeline, main
from rag_lab.config import FIXTURE_SOURCE_FILE, LEGACY_PTO_DAYS, PLANTED_DOC_NAME
from rag_lab.demo import DemoSession, build_server, demo_prompts, load_project_env
from rag_lab.generate import TextGenerator


def _session(
    tmp_path: Path,
    factory: Callable[[], TextGenerator] | None = None,
) -> DemoSession:
    pipeline = _opener(tmp_path)()
    return DemoSession(
        pipeline,
        generator_factory=factory,
        eval_record_path=tmp_path / "missing-record.json",
    )


def test_overview_keeps_the_legacy_fixture(tmp_path: Path) -> None:
    session = _session(tmp_path)
    try:
        payload = session.overview()
    finally:
        session.close()
    sources = {row["source_file"] for row in payload["corpus"]}
    assert FIXTURE_SOURCE_FILE in sources
    legacy = [row for row in payload["corpus"] if row["status"] == "legacy"]
    assert len(legacy) == 1
    assert legacy[0]["doc_name"] == PLANTED_DOC_NAME
    assert payload["chunks"] > 0
    assert any(prompt["id"] == "pto_privilege_leave" for prompt in payload["prompts"])


def test_pto_ask_shows_the_incident(tmp_path: Path) -> None:
    session = _session(tmp_path, _FactGenerator)
    prompt = demo_prompts()[0]
    assert prompt.prompt_id == "pto_privilege_leave"
    question = prompt.question
    try:
        payload = session.ask(question, generate=True)
    finally:
        session.close()
    assert payload["extractive"]["status"] == "legacy"
    assert payload["extractive"]["doc_name"] == PLANTED_DOC_NAME
    assert f"{LEGACY_PTO_DAYS} days" in payload["extractive"]["text"]
    incident = payload["incident"]
    assert incident["active"] is True
    assert incident["retrieval_found_both"] is True
    assert incident["extractive_used_legacy"] is True
    assert incident["published_retired_days"] is False
    assert incident["cites_current"] is True
    assert incident["marks_legacy_conflict"] is True
    assert payload["screen"]["has_legacy_conflict"] is True
    assert payload["dense"]
    assert payload["hybrid"]
    assert payload["rerank"]


def test_chat_scoreboard_records_gold_and_unknown_questions(tmp_path: Path) -> None:
    session = _session(tmp_path, _FactGenerator)
    pto = next(
        prompt.question
        for prompt in demo_prompts()
        if prompt.prompt_id == "pto_privilege_leave"
    )
    wifi = "What is the Wi-Fi password?"
    other = "How many sick days do I get?"
    try:
        session.ask(pto, generate=True)
        session.ask(wifi, generate=True)
        session.ask(other, generate=False)
        turns = session.session_evals()["turns"]
    finally:
        session.close()

    assert [row["question"] for row in turns] == [other, wifi, pto]

    pto_row = turns[2]
    assert pto_row["query_id"] == "pto_privilege_leave"
    assert pto_row["in_eval_set"] is True
    assert pto_row["retrieval"]["recall_at_k"] == 1.0
    assert pto_row["retrieval"]["retrieval_label"] == "data_quality_fixture"
    assert pto_row["generated"]["key_fact_correct"] is True
    assert pto_row["generated"]["conflict_handled"] is True
    assert pto_row["abstained"] is False
    assert f"{LEGACY_PTO_DAYS} days" not in pto_row["prose"]

    wifi_row = turns[1]
    assert wifi_row["query_id"] == "out_of_domain_wifi"
    assert wifi_row["retrieval"] is None
    assert wifi_row["generated"]["key_fact_correct"] is True
    assert wifi_row["abstained"] is True

    other_row = turns[0]
    assert other_row["in_eval_set"] is False
    assert other_row["query_id"] is None
    assert other_row["retrieval"] is None
    assert other_row["generated"] is None
    assert other_row["abstained"] is None


def test_non_fixture_question_skips_the_incident_banner(tmp_path: Path) -> None:
    session = _session(tmp_path)
    try:
        payload = session.ask("What is the Sexual Harassment Redressal Committee email id?")
    finally:
        session.close()
    assert payload["incident"] == {"active": False}
    assert payload["answer"]["requested"] is False


def test_empty_question_is_rejected(tmp_path: Path) -> None:
    session = _session(tmp_path)
    try:
        with pytest.raises(Exception, match="empty"):
            session.ask("   ")
    finally:
        session.close()


def test_http_serves_the_page_and_a_question(tmp_path: Path) -> None:
    pipeline = _opener(tmp_path)()
    server = build_server(
        pipeline,
        host="127.0.0.1",
        port=0,
        generator_factory=_FactGenerator,
        eval_record_path=tmp_path / "record.json",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        with urlopen(base + "/") as response:
            assert response.status == 200
            page = response.read().decode("utf-8")
        assert "Policy assistant" in page
        assert "Send" in page
        assert "search steps" in page

        with urlopen(base + "/api/overview") as response:
            overview = json.load(response)
        assert overview["documents"] == 10
        assert any(row["source_file"] == FIXTURE_SOURCE_FILE for row in overview["corpus"])

        question = overview["prompts"][0]["question"]
        body = json.dumps({"question": question, "generate": False}).encode("utf-8")
        request = Request(
            base + "/api/ask",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            asked = json.load(response)
        assert asked["extractive"]["status"] == "legacy"

        chat_body = json.dumps({"question": question}).encode("utf-8")
        chat_request = Request(
            base + "/api/chat",
            data=chat_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(chat_request) as response:
            raw = response.read().decode("utf-8")
        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
        stage_ids = [event["id"] for event in events if event["event"] == "stage"]
        assert stage_ids[0] == "dense"
        assert "rerank" in stage_ids
        assert "screen" in stage_ids
        assert events[-1]["event"] == "answer"
        assert events[-1]["answer"]["citations"]
        assert events[-1]["eval"]["query_id"] == "pto_privilege_leave"
        assert events[-1]["eval"]["generated"]["key_fact_correct"] is True
        assert f"{LEGACY_PTO_DAYS} days" not in events[-1]["answer"]["text"]

        with urlopen(base + "/api/session") as response:
            session_payload = json.load(response)
        assert session_payload["turns"][0]["question"] == question

        with pytest.raises(HTTPError) as caught:
            urlopen(base + "/api/missing")
        assert caught.value.code == 404

        record = {
            "retrieval": {
                "recall_at_k": 1.0,
                "extractive_answer_accuracy": 1.0,
                "results": [],
            }
        }
        (tmp_path / "record.json").write_text(json.dumps(record), encoding="utf-8")
        with urlopen(base + "/api/eval") as response:
            captured = json.load(response)
        assert captured["available"] is True
        assert captured["record"]["retrieval"]["recall_at_k"] == 1.0
    finally:
        server.shutdown()
        server.server_close()
        pipeline.close()


def test_load_project_env_fills_only_unset_llm_vars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "RAG_LAB_LLM_MODEL=qwen3:8b",
                "RAG_LAB_LLM_BASE_URL=http://host.docker.internal:11434/v1",
                "OPENAI_API_KEY=",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("RAG_LAB_LLM_MODEL", raising=False)
    monkeypatch.delenv("RAG_LAB_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "keep-me")
    load_project_env(env_file)
    assert os.environ["RAG_LAB_LLM_MODEL"] == "qwen3:8b"
    assert os.environ["RAG_LAB_LLM_BASE_URL"] == "http://host.docker.internal:11434/v1"
    assert os.environ["OPENAI_API_KEY"] == "keep-me"


def test_demo_command_uses_the_open_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_serve(pipeline: Pipeline, **kwargs: object) -> None:
        seen["documents"] = len(pipeline.documents)
        seen["port"] = kwargs["port"]

    monkeypatch.setattr("rag_lab.demo.serve_demo", fake_serve)
    code = main(["demo", "--port", "8765"], opener=_opener(tmp_path))
    assert code == 0
    assert seen["documents"] == 10
    assert seen["port"] == 8765
