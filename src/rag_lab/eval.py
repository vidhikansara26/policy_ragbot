"""Evaluation harness for the reranked policy retriever.

Each question has one supporting passage. The passage is not typed in as a
guess: ``bind_gold`` finds a verbatim span in exactly one file under
``data/raw``, then reads ``doc_name``, ``section``, ``version``, and
``status`` off that file and its chunk. A span that is missing or that
appears in two files is an ``EvalError``.

Two checks stay separate. Recall@K asks whether that passage is inside the
window ``search`` returns. Answer accuracy asks whether the extractive
answer — the rank-1 passage — contains the span and cites the same
document, section, and version. A gold passage at rank 3 is a recall hit
and an accuracy miss. There is no combined pass bit.

K is ``EVAL_K``, which is ``RETRIEVE_K`` (5). That is the window the
cross-encoder returns. ``RERANK_CANDIDATE_K`` (20) is only the pool that
gets rescored. Recall at 20 would treat a discarded passage as if the
caller had seen it.

``status=legacy`` is not a filter, on the gold lookup or on the retrieved
window. Human Rights Policy v1 states a 15-day Privilege Leave entitlement
that v2 does not. When that v1 passage is inside the window, the PTO row
is labeled ``data_quality_fixture``. Retrieving the planted clause is the
incident the index is supposed to surface, not a failed retrieval. Failing
to retrieve it is still a miss. Answer accuracy on that row only says the
top passage is the v1 clause from the file. It does not say 15 days is the
current entitlement. The current wording is its own question. Which of the
two an answer should have used is the next diagnosis step, not this score.

The answer is extractive. No generator is called. The rank-1 chunk text is
the answer, and it must carry doc name, section, and version.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from rag_lab.chunking import Chunk, chunk_document
from rag_lab.config import EVAL_K, LEGACY_PTO_DAYS
from rag_lab.corpus import Document
from rag_lab.exceptions import EvalError
from rag_lab.rerank import RerankedHit

logger = logging.getLogger(__name__)

_CITATION_FIELDS: tuple[str, ...] = ("doc_name", "section", "version", "status")


class Searcher(Protocol):
    """Reranked retriever. ``search`` returns the best passages first."""

    def search(self, query: str, *, k: int) -> Sequence[RerankedHit]:
        """Return up to ``k`` reranked passages.

        Raises:
            RetrievalError: If the query cannot be searched.
        """
        ...


class RetrievalLabel(StrEnum):
    """How to read a recall hit. Fixture is not a miss."""

    HIT = "hit"
    MISS = "miss"
    DATA_QUALITY_FIXTURE = "data_quality_fixture"


@dataclass(frozen=True)
class EvalCase:
    """A question plus the verbatim span that supports it.

    Document name, section, version, and status are filled from ``data/raw``
    by :func:`bind_gold`. They are not fields here, so a heading or version
    edit in the markdown becomes the label without a second copy in Python.

    Raises:
        EvalError: If the id, question, or span is blank.
    """

    query_id: str
    question: str
    span: str
    is_data_quality_fixture: bool = False

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise EvalError("Eval case query_id is empty")
        if not self.question.strip():
            raise EvalError(f"Eval case {self.query_id!r} has an empty question")
        if not self.span.strip():
            raise EvalError(f"Eval case {self.query_id!r} has an empty gold span")


@dataclass(frozen=True)
class GoldPassage:
    """One supporting passage resolved from a source file and its chunk."""

    query_id: str
    question: str
    doc_name: str
    section: str
    version: str
    status: str
    span: str
    source_file: str
    is_data_quality_fixture: bool


@dataclass(frozen=True)
class CitedAnswer:
    """Extractive answer: the top passage and the citation it must carry."""

    text: str
    doc_name: str
    section: str
    version: str
    status: str
    chunk_id: str
    score: float


@dataclass(frozen=True)
class RetrievedPassage:
    """One row of the evaluated window. Rank is 1-based, best first."""

    rank: int
    chunk_id: str
    doc_name: str
    section: str
    version: str
    status: str
    score: float
    contains_gold_span: bool


@dataclass(frozen=True)
class QueryEval:
    """Recall@K and answer accuracy for one question. They are not combined."""

    query_id: str
    question: str
    k: int
    recall_at_k: float
    answer_correct: bool
    answer: CitedAnswer
    gold: GoldPassage
    retrieved: tuple[RetrievedPassage, ...]
    retrieval_label: RetrievalLabel
    note: str

    def versions_for(self, doc_name: str) -> frozenset[str]:
        """Versions of ``doc_name`` present anywhere in the evaluated window."""
        return frozenset(row.version for row in self.retrieved if row.doc_name == doc_name)


@dataclass(frozen=True)
class EvalReport:
    """Per-query results plus the two aggregate checks.

    ``mean_recall_at_k`` averages per-query Recall@K. ``answer_accuracy`` is
    the fraction of answers that cited the gold passage. A fixture row that
    retrieved Human Rights v1 counts as recall 1. It is not dropped from
    either mean.

    Raises:
        EvalError: If ``k`` is not positive, ``results`` is empty, or
            ``query_id`` is missing.
    """

    k: int
    results: tuple[QueryEval, ...]

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise EvalError(f"Invalid k={self.k}")
        if not self.results:
            raise EvalError("Evaluation report has no queries")

    def by_id(self, query_id: str) -> QueryEval:
        """Return the single result for ``query_id``.

        Raises:
            EvalError: If ``query_id`` is missing or duplicated.
        """
        matches = [result for result in self.results if result.query_id == query_id]
        if len(matches) != 1:
            raise EvalError(f"Unknown query_id {query_id!r}")
        return matches[0]

    @property
    def mean_recall_at_k(self) -> float:
        """Mean of per-query Recall@K. Independent of answer accuracy."""
        return sum(result.recall_at_k for result in self.results) / len(self.results)

    @property
    def answer_accuracy(self) -> float:
        """Fraction of answers that matched the gold passage. Independent of recall."""
        correct = sum(1 for result in self.results if result.answer_correct)
        return correct / len(self.results)


# Spans are verbatim substrings of data/raw. bind_gold rejects a span that
# is not in exactly one file. The PTO fixture is the only legacy row: v1
# states a day count, v2 states that the current policy does not.
_EVAL_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        query_id="pto_privilege_leave",
        question="How many Privilege Leave / PTO days do I get?",
        span=f"{LEGACY_PTO_DAYS} days of Privilege Leave (PTO)",
        is_data_quality_fixture=True,
    ),
    EvalCase(
        query_id="pto_current_no_numeric_entitlement",
        question=(
            "Does the current Human Rights Policy set a numeric "
            "Privilege Leave (PTO) entitlement?"
        ),
        span="does not set a numeric Privilege Leave (PTO) entitlement",
    ),
    EvalCase(
        query_id="posh_shrc_email",
        question="What is the Sexual Harassment Redressal Committee email id?",
        span="The Sexual Harassment Redressal Committee email id is **shrc@coforge.com**.",
    ),
    EvalCase(
        query_id="posh_complaint_deadline",
        question=(
            "Within how many months must a sexual harassment complaint "
            "be reported to the SHRC?"
        ),
        span="within three months",
    ),
    EvalCase(
        query_id="whistleblower_channel",
        question="What is the dedicated whistleblower email channel?",
        span="whistleblower@coforge.com",
    ),
    EvalCase(
        query_id="whistleblower_acknowledgement",
        question=(
            "How many working days does the Compliance Officer have to "
            "acknowledge a whistleblower complaint?"
        ),
        span="acknowledge receipt within **5 working days**",
    ),
    EvalCase(
        query_id="ehs_net_zero",
        question="By when does the Environment Health and Safety policy target net zero?",
        span="Net zero by 2040",
    ),
    EvalCase(
        query_id="ehs_governance_review",
        question="Which committee reviews the EHS policy annually?",
        span="The central EHS Committee reviews the policy annually.",
    ),
    EvalCase(
        query_id="modern_slavery_training_pass_rate",
        question="What pass rate is required to complete the modern slavery training?",
        span="pass rate of 80% or over",
    ),
    EvalCase(
        query_id="nomination_independent_director_terms",
        question="How many consecutive terms may an Independent Director hold office?",
        span="two consecutive terms of up to a maximum of 5 years each",
    ),
)


def evaluation_cases() -> tuple[EvalCase, ...]:
    """Return the production query set (8 or more), including the PTO fixture.

    Raises:
        EvalError: If the set has fewer than 8 questions, duplicate ids,
            duplicate spans, or not exactly one data-quality fixture.
    """
    if len(_EVAL_CASES) < 8:
        raise EvalError(f"Eval set has {len(_EVAL_CASES)} queries; at least 8 are required")
    _require_unique(_EVAL_CASES)
    fixtures = [case for case in _EVAL_CASES if case.is_data_quality_fixture]
    if len(fixtures) != 1:
        raise EvalError(
            f"Eval set has {len(fixtures)} data-quality fixtures; exactly one is required"
        )
    return _EVAL_CASES


def bind_gold(
    cases: Sequence[EvalCase],
    documents: Sequence[Document],
) -> tuple[GoldPassage, ...]:
    """Resolve each span against ``documents`` loaded from ``data/raw``.

    ``status=legacy`` is not skipped. The stale Human Rights file is a valid
    gold source when it is the only file that contains the span.

    Raises:
        EvalError: If ``cases`` is empty, ids or questions repeat, or a span
            is not an intact passage in exactly one source file.
    """
    if not cases:
        raise EvalError("Cannot bind gold for an empty query set")
    if not documents:
        raise EvalError("Cannot bind gold against an empty corpus")
    _require_unique(cases)
    return tuple(_bind_one(case, documents) for case in cases)


def evaluate(
    retriever: Searcher,
    gold: Sequence[GoldPassage],
    *,
    k: int = EVAL_K,
) -> EvalReport:
    """Score Recall@K and answer accuracy for each gold passage.

    The retriever is asked for ``k`` passages. Legacy rows in that window
    are kept. Recall uses the whole window. Accuracy uses rank 1 only.
    A fixture passage inside the window is labeled
    :attr:`RetrievalLabel.DATA_QUALITY_FIXTURE`, not a miss.

    Raises:
        EvalError: If ``k`` is not positive, ``gold`` is empty or has
            duplicate ids, a window is empty, or a passage is missing
            doc name, section, version, or status.
        RetrievalError: If ``retriever`` rejects a question.
    """
    if k <= 0:
        raise EvalError(f"Invalid k={k}")
    if not gold:
        raise EvalError("Cannot evaluate an empty gold set")
    ids = [passage.query_id for passage in gold]
    if len(ids) != len(set(ids)):
        raise EvalError("Duplicate query_id in gold passages")

    results = tuple(_score_one(retriever, passage, k=k) for passage in gold)
    report = EvalReport(k=k, results=results)
    logger.info(
        "Eval k=%s recall@k=%.3f answer_accuracy=%.3f cases=%s",
        report.k,
        report.mean_recall_at_k,
        report.answer_accuracy,
        len(report.results),
    )
    return report


def _require_unique(cases: Sequence[EvalCase]) -> None:
    ids = [case.query_id for case in cases]
    if len(ids) != len(set(ids)):
        raise EvalError("Duplicate eval query_id")
    questions = [case.question for case in cases]
    if len(questions) != len(set(questions)):
        raise EvalError("Duplicate eval question")


def _bind_one(case: EvalCase, documents: Sequence[Document]) -> GoldPassage:
    holders = [document for document in documents if case.span in document.body]
    if len(holders) != 1:
        raise EvalError(
            f"Gold span for {case.query_id!r} matched {len(holders)} documents in "
            "data/raw. A label must be verbatim text from exactly one source file, "
            "not a guess."
        )
    document = holders[0]
    chunks = [chunk for chunk in chunk_document(document) if case.span in chunk.text]
    if not chunks:
        raise EvalError(
            f"Gold span for {case.query_id!r} is in {document.path.name} but is not "
            "intact in any chunk. Choose a span that sits inside one passage."
        )
    _require_same_citation(chunks, query_id=case.query_id, source_file=document.path.name)
    chunk = chunks[0]
    doc_name, section, version, status = _citation(chunk.metadata, chunk_id=chunk.chunk_id)
    if doc_name != document.doc_name or version != document.version or status != document.status:
        raise EvalError(
            f"Chunk {chunk.chunk_id} drifted from {document.path.name} metadata"
        )
    if case.is_data_quality_fixture and status != "legacy":
        raise EvalError(
            f"{case.query_id!r} is marked as the data-quality fixture but "
            f"{document.path.name} has status={status}"
        )
    return GoldPassage(
        query_id=case.query_id,
        question=case.question,
        doc_name=doc_name,
        section=section,
        version=version,
        status=status,
        span=case.span,
        source_file=document.path.name,
        is_data_quality_fixture=case.is_data_quality_fixture,
    )


def _require_same_citation(chunks: Sequence[Chunk], *, query_id: str, source_file: str) -> None:
    keys = {
        (
            chunk.metadata.get("doc_name", ""),
            chunk.metadata.get("section", ""),
            chunk.metadata.get("version", ""),
            chunk.metadata.get("status", ""),
        )
        for chunk in chunks
    }
    if len(keys) != 1:
        raise EvalError(
            f"Gold span for {query_id!r} lands in {len(keys)} sections of {source_file}"
        )


def _score_one(retriever: Searcher, gold: GoldPassage, *, k: int) -> QueryEval:
    hits = list(retriever.search(gold.question, k=k))[:k]
    if not hits:
        raise EvalError(
            f"Query {gold.query_id!r} returned no passages; cannot cite "
            "doc name, section, and version"
        )
    retrieved = tuple(_retrieved(hit, gold, rank=rank) for rank, hit in enumerate(hits, start=1))
    answer = _answer(hits[0])
    recall_hit = any(_is_gold(row, gold) for row in retrieved)
    answer_correct = _answer_matches(answer, gold)
    label = _label(gold, recall_hit=recall_hit)
    return QueryEval(
        query_id=gold.query_id,
        question=gold.question,
        k=k,
        recall_at_k=1.0 if recall_hit else 0.0,
        answer_correct=answer_correct,
        answer=answer,
        gold=gold,
        retrieved=retrieved,
        retrieval_label=label,
        note=_note(gold, answer, label=label, k=k),
    )


def _retrieved(hit: RerankedHit, gold: GoldPassage, *, rank: int) -> RetrievedPassage:
    doc_name, section, version, status = _citation(hit.metadata, chunk_id=hit.chunk_id)
    return RetrievedPassage(
        rank=rank,
        chunk_id=hit.chunk_id,
        doc_name=doc_name,
        section=section,
        version=version,
        status=status,
        score=hit.score,
        contains_gold_span=gold.span in hit.text,
    )


def _answer(hit: RerankedHit) -> CitedAnswer:
    doc_name, section, version, status = _citation(hit.metadata, chunk_id=hit.chunk_id)
    return CitedAnswer(
        text=hit.text,
        doc_name=doc_name,
        section=section,
        version=version,
        status=status,
        chunk_id=hit.chunk_id,
        score=hit.score,
    )


def _citation(metadata: dict[str, str], *, chunk_id: str) -> tuple[str, str, str, str]:
    values: list[str] = []
    for field in _CITATION_FIELDS:
        raw = metadata.get(field, "")
        if not isinstance(raw, str) or not raw.strip():
            raise EvalError(f"Chunk {chunk_id} is missing {field}")
        values.append(raw)
    return values[0], values[1], values[2], values[3]


def _is_gold(row: RetrievedPassage, gold: GoldPassage) -> bool:
    return (
        row.contains_gold_span
        and row.doc_name == gold.doc_name
        and row.section == gold.section
        and row.version == gold.version
        and row.status == gold.status
    )


def _answer_matches(answer: CitedAnswer, gold: GoldPassage) -> bool:
    return (
        gold.span in answer.text
        and answer.doc_name == gold.doc_name
        and answer.section == gold.section
        and answer.version == gold.version
        and answer.status == gold.status
    )


def _label(gold: GoldPassage, *, recall_hit: bool) -> RetrievalLabel:
    if not recall_hit:
        return RetrievalLabel.MISS
    if gold.is_data_quality_fixture:
        return RetrievalLabel.DATA_QUALITY_FIXTURE
    return RetrievalLabel.HIT


def _note(
    gold: GoldPassage,
    answer: CitedAnswer,
    *,
    label: RetrievalLabel,
    k: int,
) -> str:
    cited = (
        f"Answer cited {answer.doc_name} version {answer.version}, "
        f"section {answer.section!r}."
    )
    if label is RetrievalLabel.DATA_QUALITY_FIXTURE:
        return (
            f"Retrieved {gold.doc_name} version {gold.version}, "
            f"section {gold.section!r}, status={gold.status}. "
            "This hit is the data-quality fixture, not a failed retrieval. "
            f"{cited}"
        )
    if label is RetrievalLabel.HIT:
        return (
            f"Retrieved {gold.doc_name} version {gold.version}, "
            f"section {gold.section!r}. {cited}"
        )
    return (
        f"{gold.doc_name} version {gold.version} was not in the top {k}. "
        "Missing the supporting passage is a retrieval miss. "
        f"{cited}"
    )
