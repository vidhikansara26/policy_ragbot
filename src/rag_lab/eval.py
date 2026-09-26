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
current entitlement. Generation gold for that same question is a second
record: the current wording, plus a requirement to cite v2 and flag v1.

``evaluate_retrieval`` is extractive. It does not call a generator. The
rank-1 chunk text is the answer, and it must carry doc name, section, and
version. ``evaluate`` is the same function.

``evaluate_answers`` is a second scoreboard. It calls
:func:`~rag_lab.generate.generate_answer` on the same window and scores
key-fact accuracy, citation completeness, groundedness, and legacy/current
conflict handling. Those checks are not folded into ``answer_accuracy``
and they do not change Recall@K. On Privilege Leave, rank-1 accuracy means
v1 was retrieved. Key-fact accuracy means the published prose used the
current wording and did not state the planted day count.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from rag_lab.chunking import Chunk, chunk_document
from rag_lab.config import EVAL_K, LEGACY_PTO_DAYS
from rag_lab.corpus import Document
from rag_lab.exceptions import EvalError
from rag_lab.generate import ABSTAIN_TEXT, TextGenerator, generate_answer, render_answer
from rag_lab.rerank import RerankedHit
from rag_lab.safety import screen_hits

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
class AnswerCase:
    """Generation gold before :func:`bind_answer_gold` reads the source file.

    A ``must_abstain`` row has no span and is not a retrieval label. The
    Privilege Leave row sets ``legacy_conflict`` so the published answer must
    cite the current policy and flag the legacy line.

    Raises:
        EvalError: If the id or question is blank, or span and abstain disagree.
    """

    query_id: str
    question: str
    span: str = ""
    key_facts: tuple[str, ...] = ()
    must_abstain: bool = False
    legacy_conflict: bool = False

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise EvalError("Answer case query_id is empty")
        if not self.question.strip():
            raise EvalError(f"Answer case {self.query_id!r} has an empty question")
        if any(not phrase.strip() for phrase in self.key_facts):
            raise EvalError(f"Answer case {self.query_id!r} has an empty key fact")
        if self.must_abstain:
            if self.span.strip():
                raise EvalError(f"Answer case {self.query_id!r} abstains and cannot carry a span")
            if self.key_facts:
                raise EvalError(
                    f"Answer case {self.query_id!r} abstains and cannot carry key facts"
                )
            if self.legacy_conflict:
                raise EvalError(
                    f"Answer case {self.query_id!r} abstains and cannot be a legacy conflict"
                )
            return
        if not self.span.strip():
            raise EvalError(f"Answer case {self.query_id!r} has an empty gold span")


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
class AnswerGold:
    """One generation label. Abstain rows have an empty fact and no citation."""

    query_id: str
    question: str
    fact: str
    key_facts: tuple[str, ...]
    must_abstain: bool
    legacy_conflict: bool
    doc_name: str
    section: str
    version: str
    status: str
    source_file: str


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
            "Does the current Human Rights Policy set a numeric Privilege Leave (PTO) entitlement?"
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
            "Within how many months must a sexual harassment complaint be reported to the SHRC?"
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

# What a correct published answer must state, per question. The retrieval span
# stays verbatim source text for lineage; these are the load-bearing values a
# grader would check. Requiring the whole source sentence instead would score
# wording rather than correctness: "targets net zero by 2040" is a right answer
# even though the policy writes "Net zero by 2040". Matching is normalized
# (case, markdown emphasis, whitespace) and every phrase must be present.
# bind_answer_gold rejects a phrase that is not in the bound source file, so
# these cannot drift away from the corpus.
_KEY_FACTS: dict[str, tuple[str, ...]] = {
    "pto_privilege_leave": ("does not", "numeric"),
    "pto_current_no_numeric_entitlement": ("does not", "numeric"),
    "posh_shrc_email": ("shrc@coforge.com",),
    "posh_complaint_deadline": ("three months",),
    "whistleblower_channel": ("whistleblower@coforge.com",),
    "whistleblower_acknowledgement": ("5 working days",),
    "ehs_net_zero": ("net zero", "2040"),
    "ehs_governance_review": ("EHS Committee", "annual"),
    "modern_slavery_training_pass_rate": ("80%",),
    "nomination_independent_director_terms": ("two consecutive terms", "5 years"),
}

# Generation gold for the same Privilege Leave question. Retrieval keeps the
# v1 day-count span. This span is the current policy and binds to v2 only.
_NO_NUMERIC_FACT = "does not set a numeric Privilege Leave (PTO) entitlement"
# Leak-guard sentence in generate.py. It restates the v2 span.
_GUARD_PTO_FACT = "does not specify a numeric PTO allowance"
_WIFI_QUESTION = "What is the Wi-Fi password?"
_CEO_QUESTION = "What is the personal mobile number of the Coforge CEO?"

# Same sentence as generate.ABSTAIN_TEXT. A supported question that returns it
# failed to answer. An unsupported question must return it.
_UNGROUNDED_REFUSAL = ABSTAIN_TEXT
_LEGACY_CONFLICT_MARK = " (legacy conflict)"
_SECTION_NUMBER_PREFIX = re.compile(r"^\d+\.\s+")
_NONE_SOURCE = "- (none)"
# Fact matching ignores presentation. The corpus bolds its figures and a model
# does not, so emphasis, case, and line wrapping must not decide a score.
_MARKDOWN_EMPHASIS = re.compile(r"\*{1,3}|_{2}|`")
_WHITESPACE = re.compile(r"\s+")
# Two or more digits: years, percentages, day counts, and phone numbers, but not
# the single-digit section and version numbers an answer may cite about itself.
_WIDE_NUMBER = re.compile(r"\d[\d,.]*\d")
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Metadata an answer may quote about its own source without inventing anything.
_ATTRIBUTION_FIELDS: tuple[str, ...] = ("doc_name", "section", "version")


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


def answer_cases() -> tuple[AnswerCase, ...]:
    """Return generation gold, including abstentions.

    The ten retrieval questions are reused. Privilege Leave's fact is the
    current no-numeric wording, with ``legacy_conflict`` set. ``out_of_domain_wifi``
    and ``unsupported_ceo_phone`` must abstain and are not retrieval rows.

    Raises:
        EvalError: If the supported set has fewer than 10 questions, ids or
            questions repeat, or the Privilege Leave conflict row is missing.
    """
    rows: list[AnswerCase] = []
    for case in _EVAL_CASES:
        key_facts = _KEY_FACTS.get(case.query_id, ())
        if case.query_id == "pto_privilege_leave":
            rows.append(
                AnswerCase(
                    query_id=case.query_id,
                    question=case.question,
                    span=_NO_NUMERIC_FACT,
                    key_facts=key_facts,
                    legacy_conflict=True,
                )
            )
        else:
            rows.append(
                AnswerCase(
                    query_id=case.query_id,
                    question=case.question,
                    span=case.span,
                    key_facts=key_facts,
                )
            )
    rows.extend(
        (
            AnswerCase(
                query_id="out_of_domain_wifi",
                question=_WIFI_QUESTION,
                must_abstain=True,
            ),
            AnswerCase(
                query_id="unsupported_ceo_phone",
                question=_CEO_QUESTION,
                must_abstain=True,
            ),
        )
    )
    cases = tuple(rows)
    _validate_answer_cases(cases)
    return cases


def _validate_answer_cases(cases: Sequence[AnswerCase]) -> None:
    supported = [case for case in cases if not case.must_abstain]
    if len(supported) < 10:
        raise EvalError(
            f"Answer set has {len(supported)} supported questions; at least 10 are required"
        )
    ids = [case.query_id for case in cases]
    if len(ids) != len(set(ids)):
        raise EvalError("Duplicate answer query_id")
    questions = [case.question for case in cases]
    if len(questions) != len(set(questions)):
        raise EvalError("Duplicate answer question")
    conflicts = [case for case in cases if case.legacy_conflict]
    if len(conflicts) != 1 or conflicts[0].query_id != "pto_privilege_leave":
        raise EvalError("Answer set must mark pto_privilege_leave as the only legacy conflict")
    abstain_ids = {case.query_id for case in cases if case.must_abstain}
    if abstain_ids != {"out_of_domain_wifi", "unsupported_ceo_phone"}:
        raise EvalError(f"Unexpected abstain rows: {sorted(abstain_ids)}")


def bind_answer_gold(
    cases: Sequence[AnswerCase],
    documents: Sequence[Document],
) -> tuple[AnswerGold, ...]:
    """Resolve generation spans against ``documents``. Abstain rows are not bound.

    Privilege Leave binds the current no-numeric span, not the v1 day count.
    ``status=legacy`` is not skipped for any other span that happens to live
    only in a legacy file.

    Raises:
        EvalError: If ``cases`` is empty, ids repeat, a supported span is
            missing or ambiguous, or a legacy-conflict row does not bind to
            ``status=current``.
    """
    if not cases:
        raise EvalError("Cannot bind answer gold for an empty query set")
    ids = [case.query_id for case in cases]
    if len(ids) != len(set(ids)):
        raise EvalError("Duplicate query_id in answer cases")
    supported = [case for case in cases if not case.must_abstain]
    if supported and not documents:
        raise EvalError("Cannot bind answer gold against an empty corpus")
    bound: list[AnswerGold] = []
    for case in cases:
        if case.must_abstain:
            bound.append(
                AnswerGold(
                    query_id=case.query_id,
                    question=case.question,
                    fact="",
                    key_facts=(),
                    must_abstain=True,
                    legacy_conflict=False,
                    doc_name="",
                    section="",
                    version="",
                    status="",
                    source_file="",
                )
            )
            continue
        passage = _bind_one(
            EvalCase(query_id=case.query_id, question=case.question, span=case.span),
            documents,
        )
        if case.legacy_conflict and passage.status != "current":
            raise EvalError(
                f"{case.query_id!r} legacy conflict bound status={passage.status}, expected current"
            )
        key_facts = _bind_key_facts(case, documents)
        bound.append(
            AnswerGold(
                query_id=case.query_id,
                question=case.question,
                fact=_plain(case.span),
                key_facts=key_facts,
                must_abstain=False,
                legacy_conflict=case.legacy_conflict,
                doc_name=passage.doc_name,
                section=passage.section,
                version=passage.version,
                status=passage.status,
                source_file=passage.source_file,
            )
        )
    return tuple(bound)


def _bind_key_facts(
    case: AnswerCase,
    documents: Sequence[Document],
) -> tuple[str, ...]:
    """Return the phrases a published answer must state, defaulting to the span.

    Every phrase must appear in some source document, so the expected answer
    can never require wording the corpus does not contain.

    Raises:
        EvalError: If a key fact is absent from every source document.
    """
    if not case.key_facts:
        return (_plain(case.span),)
    corpus = _normalized("\n".join(document.body for document in documents))
    missing = [phrase for phrase in case.key_facts if _normalized(phrase) not in corpus]
    if missing:
        raise EvalError(f"{case.query_id!r} key facts are not in any source file: {missing}")
    return tuple(case.key_facts)


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


def evaluate_retrieval(
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


evaluate = evaluate_retrieval


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
        raise EvalError(f"Chunk {chunk.chunk_id} drifted from {document.path.name} metadata")
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
    cited = f"Answer cited {answer.doc_name} version {answer.version}, section {answer.section!r}."
    if label is RetrievalLabel.DATA_QUALITY_FIXTURE:
        return (
            f"Retrieved {gold.doc_name} version {gold.version}, "
            f"section {gold.section!r}, status={gold.status}. "
            "This hit is the data-quality fixture, not a failed retrieval. "
            f"{cited}"
        )
    if label is RetrievalLabel.HIT:
        return (
            f"Retrieved {gold.doc_name} version {gold.version}, section {gold.section!r}. {cited}"
        )
    return (
        f"{gold.doc_name} version {gold.version} was not in the top {k}. "
        "Missing the supporting passage is a retrieval miss. "
        f"{cited}"
    )


@dataclass(frozen=True)
class AnswerQueryEval:
    """One generated answer. Recall is not scored here."""

    query_id: str
    question: str
    k: int
    key_fact_correct: bool
    citation_complete: bool
    grounded: bool
    conflict_handled: bool
    published: str
    prose: str
    abstained: bool


@dataclass(frozen=True)
class AnswerEvalReport:
    """Generated-answer scoreboard. Each aggregate is its own fraction.

    Raises:
        EvalError: If ``k`` is not positive or ``results`` is empty.
    """

    k: int
    results: tuple[AnswerQueryEval, ...]

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise EvalError(f"Invalid k={self.k}")
        if not self.results:
            raise EvalError("Answer evaluation report has no queries")

    def by_id(self, query_id: str) -> AnswerQueryEval:
        """Return the single generated result for ``query_id``.

        Raises:
            EvalError: If ``query_id`` is missing or duplicated.
        """
        matches = [result for result in self.results if result.query_id == query_id]
        if len(matches) != 1:
            raise EvalError(f"Unknown query_id {query_id!r}")
        return matches[0]

    @property
    def generated_key_fact(self) -> float:
        """Fraction of answers that state the required fact or correctly abstain."""
        return _fraction([row.key_fact_correct for row in self.results], label="key-fact accuracy")

    @property
    def citation_complete(self) -> float:
        """Fraction of rows whose used Sources lines carry doc, section, and version."""
        return _fraction(
            [row.citation_complete for row in self.results],
            label="citation completeness",
        )

    @property
    def groundedness(self) -> float:
        """Fraction of rows with no claim absent from the prompt chunks."""
        return _fraction([row.grounded for row in self.results], label="groundedness")

    @property
    def conflict_handling(self) -> float:
        """Fraction of rows that cite v2 and flag v1 only when that conflict is real."""
        return _fraction([row.conflict_handled for row in self.results], label="conflict handling")


def retrieval_record(report: EvalReport) -> dict[str, Any]:
    """Serialize extractive Recall@K and answer accuracy for a JSON record."""
    return {
        "k": report.k,
        "recall_at_k": report.mean_recall_at_k,
        "extractive_answer_accuracy": report.answer_accuracy,
        "results": [
            {
                "query_id": row.query_id,
                "question": row.question,
                "recall_at_k": row.recall_at_k,
                "answer_correct": row.answer_correct,
                "retrieval_label": row.retrieval_label.value,
                "cited": {
                    "doc_name": row.answer.doc_name,
                    "section": row.answer.section,
                    "version": row.answer.version,
                    "status": row.answer.status,
                },
                "note": row.note,
            }
            for row in report.results
        ],
    }


def generated_record(report: AnswerEvalReport) -> dict[str, Any]:
    """Serialize generated-answer scores for a JSON record."""
    return {
        "k": report.k,
        "generated_key_fact": report.generated_key_fact,
        "citation_complete": report.citation_complete,
        "groundedness": report.groundedness,
        "conflict_handling": report.conflict_handling,
        "results": [
            {
                "query_id": row.query_id,
                "question": row.question,
                "key_fact_correct": row.key_fact_correct,
                "citation_complete": row.citation_complete,
                "grounded": row.grounded,
                "conflict_handled": row.conflict_handled,
                "abstained": row.abstained,
                "prose": row.prose,
            }
            for row in report.results
        ],
    }


def write_eval_record(
    path: Path,
    retrieval: EvalReport,
    generated: AnswerEvalReport | None,
) -> None:
    """Write the scoreboard(s) as JSON. ``generated`` is omitted on a retrieval-only run.

    Raises:
        EvalError: If ``path`` has no parent, or the file cannot be written.
    """
    if path.exists() and path.is_dir():
        raise EvalError(f"Eval record path is a directory: {path}")
    payload: dict[str, Any] = {
        "retrieval": retrieval_record(retrieval),
        "generated": generated_record(generated) if generated is not None else None,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise EvalError(f"Failed to write eval record to {path}") from exc


def evaluate_answers(
    retriever: Searcher,
    gold: Sequence[AnswerGold],
    generator: TextGenerator,
    *,
    k: int = EVAL_K,
) -> AnswerEvalReport:
    """Score published answers. This does not compute Recall@K.

    ``retriever`` supplies the window ``generate_answer`` sees. Metrics read
    :attr:`~rag_lab.generate.GeneratedAnswer.text` and
    :func:`~rag_lab.generate.render_answer`. A refusal of the planted day
    count cannot change a retrieval report.

    Raises:
        EvalError: If ``k`` is invalid, ids collide, a supported window is
            empty, or the published answer has no Sources block.
        GenerationError: If the generator returns unusable text.
        RetrievalError: If ``retriever`` rejects a question.
    """
    if k <= 0:
        raise EvalError(f"Invalid k={k}")
    if not gold:
        raise EvalError("Cannot evaluate an empty gold set")
    ids = [row.query_id for row in gold]
    if len(ids) != len(set(ids)):
        raise EvalError("Duplicate query_id in answer gold")

    results: list[AnswerQueryEval] = []
    for row in gold:
        hits = _search_hits(
            retriever,
            row.question,
            query_id=row.query_id,
            k=k,
            allow_empty=row.must_abstain,
        )
        generated = generate_answer(row.question, hits, generator)
        published = render_answer(generated)
        prose, sources = _split_published(published, query_id=row.query_id)
        results.append(
            _score_answer(
                row,
                hits,
                published=published,
                prose=prose,
                sources=sources,
                abstained=generated.abstained,
                k=k,
            )
        )
    report = AnswerEvalReport(k=k, results=tuple(results))
    logger.info(
        "Answer eval k=%s key_fact=%.3f citation=%.3f grounded=%.3f conflict=%.3f cases=%s",
        report.k,
        report.generated_key_fact,
        report.citation_complete,
        report.groundedness,
        report.conflict_handling,
        len(report.results),
    )
    return report


def _plain(text: str) -> str:
    return text.replace("**", "").strip()


def _normalized(text: str) -> str:
    """Casefold and drop markdown emphasis and line wrapping for fact matching."""
    return _WHITESPACE.sub(" ", _MARKDOWN_EMPHASIS.sub("", text)).strip().casefold()


def _fraction(flags: Sequence[bool], *, label: str) -> float:
    if not flags:
        raise EvalError(f"No rows to score {label}")
    return sum(1 for flag in flags if flag) / len(flags)


def _search_hits(
    retriever: Searcher,
    question: str,
    *,
    query_id: str,
    k: int,
    allow_empty: bool,
) -> list[RerankedHit]:
    hits = list(retriever.search(question, k=k))[:k]
    if not hits and not allow_empty:
        raise EvalError(
            f"Query {query_id!r} returned no passages; cannot cite doc name, section, and version"
        )
    return hits


def _split_published(published: str, *, query_id: str) -> tuple[str, list[str]]:
    marker = "\nSources:\n"
    if marker not in published:
        raise EvalError(f"Query {query_id!r} published answer has no Sources block")
    prose, _, tail = published.partition(marker)
    sources = [line for line in tail.splitlines() if line.strip()]
    return prose.strip(), sources


def _score_answer(
    gold: AnswerGold,
    hits: Sequence[RerankedHit],
    *,
    published: str,
    prose: str,
    sources: list[str],
    abstained: bool,
    k: int,
) -> AnswerQueryEval:
    return AnswerQueryEval(
        query_id=gold.query_id,
        question=gold.question,
        k=k,
        key_fact_correct=_key_fact(gold, prose),
        citation_complete=_citations_complete(sources, abstained=abstained),
        grounded=_grounded(gold, prose, hits),
        conflict_handled=_conflict_handled(gold, prose, sources, hits),
        published=published,
        prose=prose,
        abstained=abstained,
    )


def _key_fact(gold: AnswerGold, prose: str) -> bool:
    """Every required fact is stated, or the row correctly abstains.

    Matching is normalized, so a correct paraphrase counts. The planted day
    count still fails the Privilege Leave row outright.
    """
    if gold.must_abstain:
        return prose == _UNGROUNDED_REFUSAL
    if prose == _UNGROUNDED_REFUSAL:
        return False
    answer = _normalized(prose)
    if gold.fact == _NO_NUMERIC_FACT and _normalized(f"{LEGACY_PTO_DAYS} days") in answer:
        return False
    if all(_normalized(phrase) in answer for phrase in gold.key_facts):
        return True
    return gold.fact == _NO_NUMERIC_FACT and _normalized(_GUARD_PTO_FACT) in answer


def _current_text(hits: Sequence[RerankedHit]) -> str:
    return _plain("\n".join(hit.text for hit in _current_hits(hits)))


def _current_hits(hits: Sequence[RerankedHit]) -> list[RerankedHit]:
    return [hit for hit in hits if hit.metadata.get("status", "").strip() == "current"]


def _supported_text(hits: Sequence[RerankedHit]) -> str:
    """Current chunk bodies plus the attribution those chunks carry.

    An answer may name the document, section, and version it was given, so
    ``version 2.0`` and ``section 5`` are supported even though the figures
    live in metadata rather than in the passage body. Restricting support to
    bodies alone would score an answer as ungrounded for citing itself.
    """
    parts = [_current_text(hits)]
    for hit in _current_hits(hits):
        parts.extend(hit.metadata.get(field, "") for field in _ATTRIBUTION_FIELDS)
    return "\n".join(part for part in parts if part)


def _grounded(gold: AnswerGold, prose: str, hits: Sequence[RerankedHit]) -> bool:
    """Every value the answer states is in a current chunk. Not a re-check of key facts.

    Key-fact accuracy asks whether the right answer was given; groundedness asks
    whether anything was invented. An answer can be grounded and still miss the
    point, so the two are scored from different evidence: this one compares the
    answer's own identifiers and figures against the current passages.
    """
    if gold.must_abstain:
        return prose == _UNGROUNDED_REFUSAL
    if prose == _UNGROUNDED_REFUSAL:
        return False
    answer = _normalized(prose)
    planted = _normalized(f"{LEGACY_PTO_DAYS} days")
    if planted in answer and planted not in _normalized(_current_text(hits)):
        return False
    return not _unsupported_values(prose, _normalized(_supported_text(hits)))


def _unsupported_values(prose: str, supported: str) -> bool:
    """True when the answer states an email or a multi-digit figure nothing supports.

    Emails and figures are where invention shows up: a plausible-looking mailbox
    or a wrong year. Single digits are skipped because they are almost always a
    section number the answer is quoting from its own citation.
    """
    values = _EMAIL.findall(prose) + _WIDE_NUMBER.findall(prose)
    return any(_normalized(value) not in supported for value in values)


def _section_key(section: str) -> str:
    """Group ``3. Fair Wages…`` with ``5. Fair Wages…``. Citations keep the raw string."""
    return _SECTION_NUMBER_PREFIX.sub("", section.strip())


def _citations_complete(sources: list[str], *, abstained: bool) -> bool:
    """Every used source has doc, section, and version.

    A window with no used sources is complete only when the answer abstained.
    """
    if not sources or sources == [_NONE_SOURCE]:
        return abstained
    if any(line == _NONE_SOURCE for line in sources):
        return False
    return all(_parse_source_line(line) is not None for line in sources)


def _parse_source_line(line: str) -> tuple[str, str, str, bool] | None:
    """Parse ``- Doc, Section, v1.0`` with an optional legacy-conflict suffix."""
    conflict = line.endswith(_LEGACY_CONFLICT_MARK)
    body = line[: -len(_LEGACY_CONFLICT_MARK)] if conflict else line
    if not body.startswith("- "):
        return None
    body = body[2:]
    marker = ", v"
    index = body.rfind(marker)
    if index == -1:
        return None
    version = body[index + len(marker) :]
    if not version or any(character.isspace() for character in version):
        return None
    head = body[:index]
    doc_name, separator, section = head.partition(", ")
    if not separator or not doc_name.strip() or not section.strip():
        return None
    return doc_name, section, version, conflict


def _conflict_handled(
    gold: AnswerGold,
    prose: str,
    sources: Sequence[str],
    hits: Sequence[RerankedHit],
) -> bool:
    """Privilege Leave must cite v2 and flag v1. Any other row must not invent that flag.

    Scored against the screened window, not the retrieved one. A passage dropped
    below the relevance floor never reaches the prompt and cannot be cited, so
    requiring a mark for it would fail an answer for a citation it had no way to
    make.
    """
    available = screen_hits(gold.question, hits).hits
    if gold.legacy_conflict:
        return _pto_conflict(gold, prose, sources, available)
    return _window_conflict(sources, available)


def _pto_conflict(
    gold: AnswerGold,
    prose: str,
    sources: Sequence[str],
    hits: Sequence[RerankedHit],
) -> bool:
    """Do not state 15 days. Cite the current line and flag the legacy sibling."""
    if f"{LEGACY_PTO_DAYS} days" in prose:
        return False
    current_line = f"- {gold.doc_name}, {gold.section}, v{gold.version}"
    if current_line not in sources:
        return False
    for hit in hits:
        doc_name = hit.metadata.get("doc_name", "").strip()
        section = hit.metadata.get("section", "").strip()
        version = hit.metadata.get("version", "").strip()
        status = hit.metadata.get("status", "").strip()
        if status != "legacy" or doc_name != gold.doc_name or not section or not version:
            continue
        expected = f"- {doc_name}, {section}, v{version}{_LEGACY_CONFLICT_MARK}"
        if expected in sources:
            return True
    return False


def _window_conflict(sources: Sequence[str], hits: Sequence[RerankedHit]) -> bool:
    """A current/legacy sibling pair must be marked. Any other window must not be."""
    groups: dict[tuple[str, str], set[str]] = {}
    for hit in hits:
        doc_name = hit.metadata.get("doc_name", "").strip()
        section = hit.metadata.get("section", "").strip()
        status = hit.metadata.get("status", "").strip()
        groups.setdefault((doc_name, _section_key(section)), set()).add(status)
    conflict_keys = {key for key, statuses in groups.items() if {"current", "legacy"} <= statuses}
    if not conflict_keys:
        return all(_LEGACY_CONFLICT_MARK not in line for line in sources)
    for hit in hits:
        if hit.metadata.get("status", "").strip() != "legacy":
            continue
        doc_name = hit.metadata.get("doc_name", "").strip()
        section = hit.metadata.get("section", "").strip()
        key = (doc_name, _section_key(section))
        if key not in conflict_keys:
            continue
        version = hit.metadata.get("version", "").strip()
        expected = f"- {doc_name}, {section}, v{version}{_LEGACY_CONFLICT_MARK}"
        if expected not in sources:
            return False
    return True
