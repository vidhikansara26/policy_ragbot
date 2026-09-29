"""Local demo UI over the indexed policy corpus.

The page is a chatbot over the same retrieve path as ``query`` and ``answer``.
A turn streams each search stage, then the grounded answer and its citations.
``status=legacy`` is not filtered. Generation runs when the caller asks for
it and the LLM environment is configured.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rag_lab.cli import Pipeline
from rag_lab.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CURRENT_POLICY_VERSION,
    DEMO_HOST,
    DEMO_PORT,
    EMBED_MODEL,
    EVAL_K,
    FIXTURE_SOURCE_FILE,
    LEGACY_POLICY_VERSION,
    LEGACY_PTO_DAYS,
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    MIN_RERANK_SCORE,
    PLANTED_DOC_NAME,
    PROJECT_ROOT,
    RERANK_CANDIDATE_K,
    RERANK_MODEL,
    RETRIEVE_K,
    RRF_K,
    TOPIC_RERANK_SCORE,
)
from rag_lab.corpus import body_word_count
from rag_lab.eval import (
    answer_cases,
    bind_answer_gold,
    bind_gold,
    evaluate_retrieval,
    evaluation_cases,
    retrieval_record,
    score_generated_answer,
    score_retrieved_window,
)
from rag_lab.exceptions import EvalError, GenerationError, RagLabError
from rag_lab.generate import Citation, GeneratedAnswer, TextGenerator, generate_answer
from rag_lab.hybrid import HybridHit
from rag_lab.index import SearchHit
from rag_lab.rerank import RerankedHit
from rag_lab.safety import screen_hits

logger = logging.getLogger(__name__)

_UI_PATH = Path(__file__).resolve().parent / "ui" / "index.html"
_EVAL_RECORD = Path(__file__).resolve().parents[2] / "eval" / "record.json"
_MAX_BODY_BYTES = 16_384
_CITATION_FIELDS: tuple[str, ...] = ("doc_name", "section", "version", "status")


@dataclass(frozen=True)
class DemoPrompt:
    """One suggested question on the demo page."""

    prompt_id: str
    label: str
    question: str
    note: str


def demo_prompts() -> tuple[DemoPrompt, ...]:
    """Questions that walk the incident, a lexical miss, and an abstention.

    Raises:
        EvalError: If a required eval case is missing from the production set.
    """
    questions = {case.query_id: case.question for case in evaluation_cases()}
    try:
        pto = questions["pto_privilege_leave"]
        current = questions["pto_current_no_numeric_entitlement"]
        posh = questions["posh_shrc_email"]
    except KeyError as exc:
        raise EvalError(f"Demo prompt is missing eval case {exc.args[0]}") from exc
    return (
        DemoPrompt(
            prompt_id="pto_privilege_leave",
            label="Stale PTO clause",
            question=pto,
            note="Rank 1 is the retired 15-day clause. The current policy is also in the window.",
        ),
        DemoPrompt(
            prompt_id="pto_current_no_numeric_entitlement",
            label="Current wording",
            question=current,
            note="Human Rights v2 says the day count is no longer in the policy.",
        ),
        DemoPrompt(
            prompt_id="hr_complaints_email",
            label="Grievance mailbox",
            question="All.HR@coforge.com",
            note="Dense search misses this address. BM25 ranks the current grievance chunk first.",
        ),
        DemoPrompt(
            prompt_id="posh_shrc_email",
            label="POSH mailbox",
            question=posh,
            note="A current-policy fact with no version conflict.",
        ),
        DemoPrompt(
            prompt_id="out_of_domain_wifi",
            label="Out of domain",
            question="What is the Wi-Fi password?",
            note="The rerank floor drops the window, so generation abstains.",
        ),
    )


def llm_status() -> dict[str, Any]:
    """Report whether a grounded answer can be requested. The key is not included."""
    model = os.environ.get(LLM_MODEL_ENV, "").strip()
    base_url = os.environ.get(LLM_BASE_URL_ENV, "").strip()
    has_key = bool(os.environ.get(LLM_API_KEY_ENV, "").strip())
    return {
        "configured": bool(model) and (bool(base_url) or has_key),
        "model": model,
        "base_url": base_url or ("hosted OpenAI" if has_key else ""),
    }


class DemoSession:
    """JSON views over one open pipeline. Index access is serialized."""

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        generator_factory: Callable[[], TextGenerator] | None = None,
        eval_record_path: Path | None = None,
    ) -> None:
        """Bind a pipeline. ``generator_factory`` defaults to the process environment."""
        self._pipeline = pipeline
        self._generator_factory = generator_factory
        self._eval_record_path = _EVAL_RECORD if eval_record_path is None else eval_record_path
        self._lock = threading.Lock()
        self._chat_evals: list[dict[str, Any]] = []
        self._retrieval_by_question: dict[str, Any] | None = None
        self._answer_by_question: dict[str, Any] | None = None

    def close(self) -> None:
        """Release the Chroma client held by this session."""
        self._pipeline.close()

    def overview(self) -> dict[str, Any]:
        """Corpus inventory, pipeline stages, and suggested questions."""
        with self._lock:
            return {
                "documents": len(self._pipeline.documents),
                "chunks": self._pipeline.chunk_count,
                "retrieve_k": RETRIEVE_K,
                "eval_k": EVAL_K,
                "pipeline": _pipeline_stages(),
                "llm": llm_status(),
                "corpus": self._corpus_rows(),
                "prompts": [_prompt_payload(prompt) for prompt in demo_prompts()],
                "eval_record_present": self._eval_record_path.is_file(),
            }

    def ask(self, question: str, *, k: int = RETRIEVE_K, generate: bool = False) -> dict[str, Any]:
        """Run dense, fusion, and rerank, then optionally generate.

        Raises:
            EvalError: If ``k`` is outside 1..``RERANK_CANDIDATE_K``, the
                question is empty, a hit is missing citation metadata, or a
                search stage fails.
        """
        result: dict[str, Any] = {
            "question": question.strip(),
            "k": k,
            "documents": 0,
            "chunks": 0,
            "dense": [],
            "hybrid": [],
            "rerank": [],
            "extractive": {},
            "screen": {
                "abstain": True,
                "has_legacy_conflict": False,
                "floor": MIN_RERANK_SCORE,
                "band": "none",
                "hits": [],
            },
            "answer": {"requested": False, "available": False, "reason": ""},
            "incident": {"active": False},
        }
        for event in self.iter_chat(question, k=k, generate=generate):
            kind = event.get("event")
            if kind == "error":
                raise EvalError(str(event.get("error", "Demo request failed")))
            if kind == "stage" and event.get("status") == "done":
                stage_id = str(event.get("id", ""))
                hits = event.get("hits", [])
                if stage_id == "dense":
                    result["dense"] = hits
                elif stage_id == "hybrid":
                    result["hybrid"] = hits
                elif stage_id == "rerank":
                    result["rerank"] = hits
                elif stage_id == "screen":
                    result["screen"] = event["screen"]
            elif kind == "answer":
                result["question"] = event["question"]
                result["k"] = event["k"]
                result["documents"] = event["documents"]
                result["chunks"] = event["chunks"]
                result["extractive"] = event["extractive"]
                result["answer"] = event["answer"]
                result["incident"] = event["incident"]
        if not result["rerank"]:
            raise EvalError("Query returned no passages")
        return result

    def iter_chat(
        self,
        question: str,
        *,
        k: int = RETRIEVE_K,
        generate: bool = True,
    ) -> Generator[dict[str, Any], None, None]:
        """Yield search stages, then one answer event.

        The first event is emitted before dense search runs, so a client can
        show that step while retrieval is still working. ``status=legacy``
        stays in every stage.

        Raises:
            EvalError: If ``k`` or ``question`` is invalid. Raised before the
                first event. A search failure after that is an ``error`` event.
        """
        if k <= 0 or k > RERANK_CANDIDATE_K:
            raise EvalError(f"Invalid k={k}; expected 1..{RERANK_CANDIDATE_K}")
        cleaned = question.strip()
        if not cleaned:
            raise EvalError("Question text is empty")
        with self._lock:
            yield from self._chat_events(cleaned, k=k, generate=generate)

    def captured_eval(self) -> dict[str, Any]:
        """Return the committed eval record, or ``available: false`` when it is absent.

        Raises:
            EvalError: If the file exists but is not a JSON object.
        """
        path = self._eval_record_path
        if not path.is_file():
            return {"available": False, "reason": f"No eval record at {path.name}"}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvalError(f"Cannot read eval record {path}") from exc
        if not isinstance(loaded, dict):
            raise EvalError(f"Eval record {path} is not a JSON object")
        return {"available": True, "source": "captured", "record": loaded}

    def live_retrieval_eval(self) -> dict[str, Any]:
        """Score Recall@K on the open index. Does not call a language model.

        Raises:
            EvalError: If gold cannot be bound or a hit is missing citation metadata.
        """
        with self._lock:
            report = evaluate_retrieval(
                self._pipeline.retriever,
                bind_gold(evaluation_cases(), self._pipeline.documents),
                k=EVAL_K,
            )
        return {"available": True, "source": "live", "retrieval": retrieval_record(report)}

    def session_evals(self) -> dict[str, Any]:
        """Return chat questions scored after each answer, newest first."""
        with self._lock:
            return {"turns": list(self._chat_evals)}

    def _record_chat_eval(
        self,
        question: str,
        hits: Sequence[RerankedHit],
        answer: Mapping[str, Any],
        generated: GeneratedAnswer | None,
        *,
        k: int,
    ) -> dict[str, Any]:
        """Score ``question`` against the eval set and keep the row for the scoreboard.

        A question that is not in the eval set is still listed. Retrieval gold
        scores Recall@K. Generation gold scores the published answer.
        """
        retrieval_gold, answer_gold = self._gold_maps()
        passage = retrieval_gold.get(question)
        label = answer_gold.get(question)
        query_id = label.query_id if label is not None else (passage.query_id if passage else None)
        retrieval: dict[str, Any] | None = None
        if passage is not None:
            scored = score_retrieved_window(passage, hits, k=k)
            retrieval = {
                "recall_at_k": scored.recall_at_k,
                "answer_correct": scored.answer_correct,
                "retrieval_label": scored.retrieval_label.value,
            }
        generated_scores: dict[str, Any] | None = None
        if label is not None and generated is not None:
            graded = score_generated_answer(label, hits, generated, k=k)
            generated_scores = {
                "key_fact_correct": graded.key_fact_correct,
                "citation_complete": graded.citation_complete,
                "grounded": graded.grounded,
                "conflict_handled": graded.conflict_handled,
            }
        top: dict[str, Any] | None = None
        if hits:
            hit = hits[0]
            top = {
                "doc_name": str(hit.metadata.get("doc_name", "")),
                "section": str(hit.metadata.get("section", "")),
                "version": str(hit.metadata.get("version", "")),
                "status": str(hit.metadata.get("status", "")),
                "score": hit.score,
            }
        abstained = answer.get("abstained") if answer.get("available") is True else None
        prose = str(answer.get("text", "")) if answer.get("available") is True else ""
        row = {
            "question": question,
            "query_id": query_id,
            "in_eval_set": query_id is not None,
            "abstained": abstained,
            "prose": prose,
            "top": top,
            "retrieval": retrieval,
            "generated": generated_scores,
        }
        self._chat_evals.insert(0, row)
        return row

    def _gold_maps(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Bind eval questions once. Keys are the question strings."""
        if self._retrieval_by_question is None or self._answer_by_question is None:
            documents = self._pipeline.documents
            self._retrieval_by_question = {
                row.question: row for row in bind_gold(evaluation_cases(), documents)
            }
            self._answer_by_question = {
                row.question: row for row in bind_answer_gold(answer_cases(), documents)
            }
        return self._retrieval_by_question, self._answer_by_question

    def _corpus_rows(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for chunk in self._pipeline.index.list_chunks():
            source = chunk.metadata.get("source_file", "")
            counts[source] = counts.get(source, 0) + 1
        rows: list[dict[str, Any]] = []
        for document in self._pipeline.documents:
            source = document.path.name
            rows.append(
                {
                    "source_file": source,
                    "doc_name": document.doc_name,
                    "version": document.version,
                    "status": document.status,
                    "role": document.role,
                    "words": body_word_count(document.body),
                    "sections": sum(
                        1 for line in document.body.splitlines() if line.startswith("## ")
                    ),
                    "chunks": counts.get(source, 0),
                    "fixture": source == FIXTURE_SOURCE_FILE,
                }
            )
        rows.sort(key=lambda row: (0 if row["status"] == "legacy" else 1, str(row["doc_name"])))
        return rows

    def _maybe_generate(
        self,
        question: str,
        hits: Sequence[RerankedHit],
        *,
        generate: bool,
    ) -> tuple[dict[str, Any], GeneratedAnswer | None]:
        """Return the answer payload and, when generation ran, the scored object."""
        if not generate:
            return {
                "requested": False,
                "available": False,
                "reason": "Generation was not requested.",
            }, None
        status = llm_status()
        if self._generator_factory is None and not status["configured"]:
            return {
                "requested": True,
                "available": False,
                "reason": (
                    f"Set {LLM_MODEL_ENV} and either {LLM_BASE_URL_ENV} or {LLM_API_KEY_ENV}, "
                    "then restart the demo."
                ),
            }, None
        factory = self._generator_factory
        if factory is None:
            from rag_lab.generate import generator_from_env

            factory = generator_from_env
        try:
            generated = generate_answer(question, hits, factory())
        except GenerationError as exc:
            return {"requested": True, "available": False, "reason": str(exc)}, None
        return _answer_payload(generated), generated

    def _chat_events(
        self,
        question: str,
        *,
        k: int,
        generate: bool,
    ) -> Iterator[dict[str, Any]]:
        yield _stage(
            "dense",
            "running",
            "Dense search",
            summary="Embedding the question with MiniLM.",
        )
        try:
            dense_hits = self._pipeline.index.search(question, k=k)
        except RagLabError as exc:
            yield {"event": "error", "error": str(exc)}
            return
        dense = [
            _labeled(_dense_row(hit, rank), f"distance {hit.distance:.3f}")
            for rank, hit in enumerate(dense_hits, start=1)
        ]
        yield _stage(
            "dense",
            "done",
            "Dense search",
            summary=_lead_summary("Top dense passage", dense),
            hits=dense,
        )

        yield _stage(
            "hybrid",
            "running",
            "BM25 + reciprocal rank fusion",
            summary="Fusing the dense list with a lexical search.",
        )
        try:
            hybrid_hits = self._pipeline.retriever.hybrid.search(question, k=k)
        except RagLabError as exc:
            yield {"event": "error", "error": str(exc)}
            return
        hybrid = [
            _labeled(_hybrid_row(hit, rank), f"rrf {hit.rrf_score:.4f}")
            for rank, hit in enumerate(hybrid_hits, start=1)
        ]
        yield _stage(
            "hybrid",
            "done",
            "BM25 + reciprocal rank fusion",
            summary=_fusion_summary(hybrid),
            hits=hybrid,
        )

        yield _stage(
            "rerank",
            "running",
            "Cross-encoder rerank",
            summary="Scoring each query and passage together.",
        )
        try:
            reranked = self._pipeline.retriever.search(question, k=k)
        except RagLabError as exc:
            yield {"event": "error", "error": str(exc)}
            return
        if not reranked:
            yield {"event": "error", "error": "Query returned no passages"}
            return
        rerank = [
            _labeled(_rerank_row(hit, rank), f"score {hit.score:.3f}")
            for rank, hit in enumerate(reranked, start=1)
        ]
        yield _stage(
            "rerank",
            "done",
            "Cross-encoder rerank",
            summary=_lead_summary("Cross-encoder placed first", rerank),
            hits=rerank,
        )

        yield _stage(
            "screen",
            "running",
            "Safety screen",
            summary="Dropping weak scores and putting current policy ahead of legacy.",
        )
        try:
            screened = screen_hits(question, reranked)
        except RagLabError as exc:
            yield {"event": "error", "error": str(exc)}
            return
        screen_hits_rows = [
            _labeled(_rerank_row(hit, rank), f"score {hit.score:.3f}")
            for rank, hit in enumerate(screened.hits, start=1)
        ]
        screen = {
            "abstain": screened.abstain,
            "has_legacy_conflict": screened.has_legacy_conflict,
            "floor": MIN_RERANK_SCORE,
            "band": screened.band,
            "hits": screen_hits_rows,
        }
        yield _stage(
            "screen",
            "done",
            "Safety screen",
            summary=_screen_summary(screen),
            hits=screen_hits_rows,
            screen=screen,
        )

        if generate:
            yield _stage(
                "generate",
                "running",
                "Writing the answer",
                summary="The model sees only the screened passages.",
            )
        answer, generated = self._maybe_generate(question, reranked, generate=generate)
        recorded = self._record_chat_eval(question, reranked, answer, generated, k=k)
        if generate:
            reason = str(answer.get("reason", ""))
            summary = (
                reason
                if not answer.get("available")
                else ("Citations are built from the passages, not from the model.")
            )
            yield _stage("generate", "done", "Writing the answer", summary=summary)
        yield {
            "event": "answer",
            "question": question,
            "k": k,
            "documents": len(self._pipeline.documents),
            "chunks": self._pipeline.chunk_count,
            "extractive": rerank[0],
            "answer": answer,
            "incident": _incident(question, reranked, answer),
            "eval": recorded,
        }


class DemoServer(ThreadingHTTPServer):
    """HTTP server that serves one :class:`DemoSession`."""

    def __init__(self, server_address: tuple[str, int], session: DemoSession) -> None:
        self.session = session
        super().__init__(server_address, DemoHandler)


class DemoHandler(BaseHTTPRequestHandler):
    """GET the page and the scoreboard. POST a question or a live retrieval eval.

    ``/api/chat`` streams newline-delimited JSON: search stages, then the answer.
    """

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/":
                self._send_html()
            elif path == "/api/overview":
                self._send_json(200, self._session().overview())
            elif path == "/api/eval":
                self._send_json(200, self._session().captured_eval())
            elif path == "/api/session":
                self._send_json(200, self._session().session_evals())
            else:
                self._send_json(404, {"error": f"Unknown path {path}"})
        except RagLabError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception:
            logger.exception("Demo GET failed")
            self._send_json(500, {"error": "Demo request failed"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            session = self._session()
            if path == "/api/ask":
                question, k, generate = _parse_chat_body(payload, default_generate=False)
                self._send_json(200, session.ask(question, k=k, generate=generate))
            elif path == "/api/chat":
                question, k, generate = _parse_chat_body(payload, default_generate=True)
                events = session.iter_chat(question, k=k, generate=generate)
                try:
                    first = next(events)
                except RagLabError:
                    events.close()
                    raise
                except Exception:
                    events.close()
                    raise
                self._send_ndjson(_prefixed(first, events))
            elif path == "/api/eval":
                self._send_json(200, session.live_retrieval_eval())
            else:
                self._send_json(404, {"error": f"Unknown path {path}"})
        except RagLabError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception:
            logger.exception("Demo POST failed")
            self._send_json(500, {"error": "Demo request failed"})

    def log_message(self, fmt: str, *args: object) -> None:
        logger.info("demo %s", fmt % args)

    def _session(self) -> DemoSession:
        server = self.server
        if not isinstance(server, DemoServer):
            raise RagLabError("Demo handler is not bound to a demo server")
        return server.session

    def _read_json(self) -> dict[str, Any]:
        length_header = self.headers.get("Content-Length", "")
        try:
            length = int(length_header)
        except ValueError as exc:
            raise EvalError("Content-Length is required") from exc
        if length < 0 or length > _MAX_BODY_BYTES:
            raise EvalError(f"Request body must be 0..{_MAX_BODY_BYTES} bytes")
        raw = self.rfile.read(length)
        try:
            loaded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvalError("Request body is not JSON") from exc
        if not isinstance(loaded, dict):
            raise EvalError("Request body must be a JSON object")
        return loaded

    def _send_html(self) -> None:
        if not _UI_PATH.is_file():
            raise RagLabError(f"Demo page is missing at {_UI_PATH}")
        body = _UI_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_ndjson(self, events: Iterator[dict[str, Any]]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for event in events:
                self._write_chunk((json.dumps(event) + "\n").encode("utf-8"))
        except RagLabError as exc:
            self._write_chunk((json.dumps({"event": "error", "error": str(exc)}) + "\n").encode())
        except Exception:
            logger.exception("Demo chat stream failed")
            message = json.dumps({"event": "error", "error": "Demo request failed"})
            self._write_chunk((message + "\n").encode("utf-8"))
        self._write_chunk(b"")

    def _write_chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_server(
    pipeline: Pipeline,
    *,
    host: str = DEMO_HOST,
    port: int = DEMO_PORT,
    generator_factory: Callable[[], TextGenerator] | None = None,
    eval_record_path: Path | None = None,
) -> DemoServer:
    """Bind a demo server. ``port`` 0 asks the OS for a free port.

    Raises:
        RagLabError: If ``host`` is blank, ``port`` is out of range, or the
            address cannot be bound.
    """
    if not host.strip():
        raise RagLabError("Demo host is empty")
    if port < 0 or port > 65535:
        raise RagLabError(f"Invalid demo port {port}")
    session = DemoSession(
        pipeline,
        generator_factory=generator_factory,
        eval_record_path=eval_record_path,
    )
    try:
        return DemoServer((host, port), session)
    except OSError as exc:
        raise RagLabError(f"Cannot bind demo server at {host}:{port}") from exc


def load_project_env(path: Path | None = None) -> None:
    """Fill unset LLM settings from the project ``.env`` file.

    Existing environment variables win. A missing file is ignored. Values are
    not logged.
    """
    env_path = PROJECT_ROOT / ".env" if path is None else path
    if not env_path.is_file():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RagLabError(f"Cannot read {env_path}") from exc
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


def serve_demo(
    pipeline: Pipeline,
    *,
    host: str = DEMO_HOST,
    port: int = DEMO_PORT,
    generator_factory: Callable[[], TextGenerator] | None = None,
    eval_record_path: Path | None = None,
) -> None:
    """Serve the demo until the process is interrupted.

    Reads ``.env`` for ``RAG_LAB_LLM_MODEL`` and ``RAG_LAB_LLM_BASE_URL`` when
    those variables are not already set.

    Raises:
        RagLabError: If the server cannot bind or ``.env`` cannot be read.
    """
    load_project_env()
    server = build_server(
        pipeline,
        host=host,
        port=port,
        generator_factory=generator_factory,
        eval_record_path=eval_record_path,
    )
    raw_host, bound_port = server.server_address[:2]
    bound_host = raw_host.decode("utf-8") if isinstance(raw_host, bytes) else raw_host
    line = (
        f"Policy demo http://{bound_host}:{bound_port}\n"
        f"Indexed {pipeline.chunk_count} chunks from {len(pipeline.documents)} documents.\n"
    )
    sys.stdout.write(line)
    sys.stdout.flush()
    logger.info("Demo listening on http://%s:%s", bound_host, bound_port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _pipeline_stages() -> list[dict[str, str]]:
    return [
        {
            "id": "chunk",
            "label": "Chunk",
            "detail": f"{CHUNK_SIZE} character budget, {CHUNK_OVERLAP} overlap on windows",
        },
        {"id": "dense", "label": "MiniLM dense", "detail": EMBED_MODEL},
        {"id": "fusion", "label": "BM25 + RRF", "detail": f"Reciprocal rank fusion, k={RRF_K}"},
        {"id": "rerank", "label": "Cross-encoder", "detail": RERANK_MODEL},
        {
            "id": "safety",
            "label": "Safety screen",
            "detail": (
                f"Factoid floor {MIN_RERANK_SCORE:.0f}; "
                f"on-topic band from {TOPIC_RERANK_SCORE:.0f} when nothing clears it; "
                "current before legacy"
            ),
        },
        {
            "id": "answer",
            "label": "Grounded answer",
            "detail": "Citations built in code, including a legacy-conflict mark",
        },
    ]


def _parse_chat_body(
    payload: Mapping[str, Any],
    *,
    default_generate: bool,
) -> tuple[str, int, bool]:
    """Read question, k, and generate from a chat or ask body.

    Raises:
        EvalError: If a field has the wrong type.
    """
    question = payload.get("question", "")
    if not isinstance(question, str):
        raise EvalError("question must be a string")
    k = payload.get("k", RETRIEVE_K)
    if not isinstance(k, int) or isinstance(k, bool):
        raise EvalError("k must be an integer")
    generate = payload.get("generate", default_generate)
    if not isinstance(generate, bool):
        raise EvalError("generate must be a boolean")
    return question, k, generate


def _prefixed(
    first: dict[str, Any],
    events: Generator[dict[str, Any], None, None],
) -> Iterator[dict[str, Any]]:
    try:
        yield first
        yield from events
    finally:
        events.close()


def _stage(
    stage_id: str,
    status: str,
    title: str,
    *,
    summary: str = "",
    hits: list[dict[str, Any]] | None = None,
    screen: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event": "stage",
        "id": stage_id,
        "status": status,
        "title": title,
        "summary": summary,
        "hits": [] if hits is None else hits,
    }
    if screen is not None:
        event["screen"] = screen
    return event


def _labeled(row: dict[str, Any], score_label: str) -> dict[str, Any]:
    row["score_label"] = score_label
    return row


def _lead_summary(prefix: str, hits: Sequence[Mapping[str, Any]]) -> str:
    if not hits:
        return f"{prefix}: no passages."
    hit = hits[0]
    return f"{prefix}: {hit['doc_name']} v{hit['version']} ({hit['status']})."


def _fusion_summary(hits: Sequence[Mapping[str, Any]]) -> str:
    if not hits:
        return "Fusion returned no passages."
    hit = hits[0]
    dense_rank = hit.get("dense_rank")
    bm25_rank = hit.get("bm25_rank")
    dense = "not in the dense list" if dense_rank is None else f"dense rank {dense_rank}"
    bm25 = "not in the BM25 list" if bm25_rank is None else f"BM25 rank {bm25_rank}"
    return f"Fused rank 1 is {hit['doc_name']} v{hit['version']} ({dense}, {bm25})."


def _screen_summary(screen: Mapping[str, Any]) -> str:
    hits = screen.get("hits", [])
    count = len(hits) if isinstance(hits, list) else 0
    if screen.get("abstain") is True:
        return "No passage cleared the score floor, so the answer will abstain."
    if screen.get("has_legacy_conflict") is True:
        return f"Kept {count} passages. Current policy is ordered ahead of the legacy conflict."
    if screen.get("band") == "topic":
        return f"Kept {count} passages in the on-topic score band."
    return f"Kept {count} passages above the score floor."


def _prompt_payload(prompt: DemoPrompt) -> dict[str, str]:
    return {
        "id": prompt.prompt_id,
        "label": prompt.label,
        "question": prompt.question,
        "note": prompt.note,
    }


def _citation(hit: SearchHit | HybridHit | RerankedHit) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in _CITATION_FIELDS:
        raw = hit.metadata.get(field, "")
        if not isinstance(raw, str) or not raw.strip():
            raise EvalError(f"Chunk {hit.chunk_id} is missing {field}")
        values[field] = raw
    role = hit.metadata.get("role", "")
    values["role"] = role if isinstance(role, str) else ""
    return values


def _dense_row(hit: SearchHit, rank: int) -> dict[str, Any]:
    row = _citation(hit)
    row.update(rank=rank, chunk_id=hit.chunk_id, text=hit.text, distance=hit.distance)
    return row


def _hybrid_row(hit: HybridHit, rank: int) -> dict[str, Any]:
    row = _citation(hit)
    row.update(
        rank=rank,
        chunk_id=hit.chunk_id,
        text=hit.text,
        rrf_score=hit.rrf_score,
        dense_rank=hit.dense_rank,
        bm25_rank=hit.bm25_rank,
    )
    return row


def _rerank_row(hit: RerankedHit, rank: int) -> dict[str, Any]:
    row = _citation(hit)
    row.update(
        rank=rank,
        chunk_id=hit.chunk_id,
        text=hit.text,
        score=hit.score,
        rrf_score=hit.rrf_score,
        dense_rank=hit.dense_rank,
        bm25_rank=hit.bm25_rank,
    )
    return row


def _answer_payload(answer: GeneratedAnswer) -> dict[str, Any]:
    return {
        "requested": True,
        "available": True,
        "text": answer.text,
        "abstained": answer.abstained,
        "has_legacy_conflict": answer.has_legacy_conflict,
        "citations": [_citation_payload(item) for item in answer.citations],
    }


def _citation_payload(item: Citation) -> dict[str, Any]:
    return {
        "doc_name": item.doc_name,
        "section": item.section,
        "version": item.version,
        "status": item.status,
        "legacy_conflict": item.legacy_conflict,
    }


def _incident(
    question: str,
    hits: Sequence[RerankedHit],
    answer: Mapping[str, Any],
) -> dict[str, Any]:
    pto = next(prompt for prompt in demo_prompts() if prompt.prompt_id == "pto_privilege_leave")
    if question != pto.question:
        return {"active": False}
    versions = {
        hit.metadata.get("version")
        for hit in hits
        if hit.metadata.get("doc_name") == PLANTED_DOC_NAME
    }
    found_both = LEGACY_POLICY_VERSION in versions and CURRENT_POLICY_VERSION in versions
    extractive_legacy = hits[0].metadata.get("status") == "legacy"
    published_days: bool | None = None
    cites_current: bool | None = None
    marks_conflict: bool | None = None
    if answer.get("available") is True:
        text = str(answer.get("text", ""))
        published_days = f"{LEGACY_PTO_DAYS} days" in text
        citations = answer.get("citations", [])
        cites_current = False
        marks_conflict = False
        if isinstance(citations, list):
            for item in citations:
                if not isinstance(item, dict):
                    continue
                if (
                    item.get("doc_name") == PLANTED_DOC_NAME
                    and item.get("version") == CURRENT_POLICY_VERSION
                    and item.get("legacy_conflict") is False
                ):
                    cites_current = True
                if item.get("legacy_conflict") is True:
                    marks_conflict = True
    return {
        "active": True,
        "retrieval_found_both": found_both,
        "extractive_used_legacy": extractive_legacy,
        "published_retired_days": published_days,
        "cites_current": cites_current,
        "marks_legacy_conflict": marks_conflict,
    }
