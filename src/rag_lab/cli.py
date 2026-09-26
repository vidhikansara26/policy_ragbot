"""Command-line runner for query, compare, answer, and the eval scoreboard.

``query`` loads ``data/raw``, chunks, embeds, stores, fuses dense and BM25
with reciprocal rank fusion, reranks, and prints the returned chunks plus
the extractive answer. The answer is the rank-1 passage. It carries doc
name, section, and version. ``status=legacy`` is not filtered.

``compare`` prints dense-only and RRF hits on the same rank rows. It does
not call the cross-encoder. ``dense_rank``, ``bm25_rank``, and ``rrf_score``
come from the fused hit. ``status=legacy`` is not filtered.

``answer`` runs the same retrieve path, then calls
:func:`~rag_lab.generate.generate_answer` with
:func:`~rag_lab.generate.generator_from_env`. The model sees only that
window. Citations are built in code, including a legacy-conflict mark when
current and legacy of the same policy disagree. ``query`` stays extractive.

``eval`` runs the extractive pipeline over the production question set and
prints Recall@K and extractive answer accuracy on separate lines. It then
calls :func:`~rag_lab.eval.evaluate_answers` and prints key-fact accuracy,
citation completeness, groundedness, and conflict handling. Those four
lines are not Recall@K. K is ``EVAL_K``. The Privilege Leave retrieval row
is labeled ``data_quality_fixture`` when Human Rights Policy v1 is in the
window. Extractive accuracy on that row still means the rank-1 chunk is v1.

Model ids, K, and the Chroma directory stay in ``rag_lab.config``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_lab.chunking import chunk_corpus
from rag_lab.config import CHROMA_DIR, EVAL_K, RETRIEVE_K
from rag_lab.corpus import Document, load_corpus
from rag_lab.embeddings import TextEmbedder
from rag_lab.eval import (
    AnswerEvalReport,
    EvalReport,
    RetrievalLabel,
    answer_cases,
    bind_answer_gold,
    bind_gold,
    evaluate_answers,
    evaluate_retrieval,
    evaluation_cases,
)
from rag_lab.exceptions import DataLoadError, EvalError, RagLabError
from rag_lab.generate import TextGenerator, generate_answer, generator_from_env, render_answer
from rag_lab.hybrid import HybridHit, HybridRetriever
from rag_lab.index import PolicyIndex, SearchHit
from rag_lab.rerank import PairScorer, RerankedHit, RerankingRetriever

logger = logging.getLogger(__name__)

_CITATION_FIELDS: tuple[str, ...] = ("doc_name", "section", "version", "status")


@dataclass
class Pipeline:
    """An open index plus the reranking retriever built from one corpus load."""

    index: PolicyIndex
    retriever: RerankingRetriever
    documents: tuple[Document, ...]
    chunk_count: int

    def close(self) -> None:
        """Release the Chroma client.

        Raises:
            IndexingError: If the client cannot be closed.
        """
        self.index.close()


def build_pipeline(
    persist_directory: Path,
    documents: Sequence[Document],
    *,
    embedder: TextEmbedder | None = None,
    scorer: PairScorer | None = None,
) -> Pipeline:
    """Chunk ``documents``, index them, and return a reranking retriever.

    Legacy documents stay in the index. ``embedder=None`` and ``scorer=None``
    use MiniLM and the MS MARCO cross-encoder.

    Raises:
        DataLoadError: If ``documents`` is empty.
        ChunkingError: If the corpus produces no chunks.
        IndexingError: If Chroma or the embedder fails.
        RetrievalError: If BM25 cannot be built from the stored rows.
    """
    if not documents:
        raise DataLoadError("Cannot index an empty corpus")
    loaded = tuple(documents)
    chunks = chunk_corpus(list(loaded))
    index = PolicyIndex(persist_directory, embedder=embedder)
    try:
        retriever = RerankingRetriever(HybridRetriever(index), scorer=scorer)
        retriever.upsert(chunks)
    except Exception:
        index.close()
        raise
    logger.info("CLI indexed %s chunks from %s documents", len(chunks), len(loaded))
    return Pipeline(
        index=index,
        retriever=retriever,
        documents=loaded,
        chunk_count=len(chunks),
    )


def open_default_pipeline() -> Pipeline:
    """Index ``data/raw`` into ``CHROMA_DIR`` with the production models.

    Raises:
        DataLoadError: If the corpus directory is missing or empty.
        IndexingError: If embedding or Chroma fails.
        RetrievalError: If the sparse index cannot be built.
    """
    return build_pipeline(CHROMA_DIR, load_corpus())


def render_query(
    question: str,
    hits: Sequence[RerankedHit],
    *,
    documents: int,
    chunk_count: int,
) -> str:
    """Format the retrieved chunks and the cited rank-1 answer.

    Raises:
        EvalError: If ``hits`` is empty or a hit is missing doc name, section,
            version, or status.
    """
    if not hits:
        raise EvalError("Query returned no passages; cannot cite doc name, section, and version")
    cited = [_citation(hit) for hit in hits]
    top = cited[0]
    lines = [
        f"Query: {question.strip()}",
        f"Indexed {chunk_count} chunks from {documents} documents.",
        (
            "Pipeline: chunk → MiniLM dense → Chroma → BM25 + reciprocal rank fusion"
            " → cross-encoder rerank."
        ),
        f"Returned {len(hits)} chunks.",
        "",
        "Chunks",
    ]
    for rank, (hit, meta) in enumerate(zip(hits, cited, strict=True), start=1):
        lines.append(
            f"{rank}. score={hit.score:.3f}  {meta['doc_name']}  "
            f"v{meta['version']}  {meta['status']}  section={meta['section']}"
        )
        lines.append(_indent(hit.text))
        lines.append("")
    lines.append("Answer")
    lines.append(
        f"{top['doc_name']}, section {top['section']!r}, "
        f"version {top['version']} (status={top['status']})"
    )
    lines.append(_indent(hits[0].text))
    return "\n".join(lines).rstrip() + "\n"


def render_compare(
    question: str,
    dense_hits: Sequence[SearchHit],
    hybrid_hits: Sequence[HybridHit],
    *,
    k: int,
) -> str:
    """Format dense-only and RRF hits on the same rank rows.

    Row ``n`` is dense rank ``n`` beside fused rank ``n``. ``dense_rank``,
    ``bm25_rank``, and ``rrf_score`` describe the fused hit. A missing rank
    prints as ``-``. ``status=legacy`` is not dropped.

    Raises:
        EvalError: If ``k`` is not positive or a hit is missing doc name,
            section, version, or status.
    """
    if k <= 0:
        raise EvalError(f"Invalid k={k}")
    dense = [_cite_label(hit) for hit in list(dense_hits)[:k]]
    hybrid = list(hybrid_hits)[:k]
    hybrid_labels = [_cite_label(hit) for hit in hybrid]
    dense_width = max(len("dense"), *(len(label) for label in dense))
    hybrid_width = max(len("hybrid"), *(len(label) for label in hybrid_labels))
    lines = [
        f"Query: {question.strip()}",
        f"k={k}",
        (
            f"{'rank':<4}  {'dense':<{dense_width}}  {'hybrid':<{hybrid_width}}  "
            "dense_rank  bm25_rank  rrf"
        ),
    ]
    for rank in range(1, k + 1):
        dense_cell = dense[rank - 1] if rank <= len(dense) else "-"
        if rank <= len(hybrid):
            hit = hybrid[rank - 1]
            hybrid_cell = hybrid_labels[rank - 1]
            dense_rank = _rank_cell(hit.dense_rank)
            bm25_rank = _rank_cell(hit.bm25_rank)
            rrf = f"{hit.rrf_score:.6f}"
        else:
            hybrid_cell = "-"
            dense_rank = "-"
            bm25_rank = "-"
            rrf = "-"
        lines.append(
            f"{rank:<4}  {dense_cell:<{dense_width}}  {hybrid_cell:<{hybrid_width}}  "
            f"{dense_rank:<10}  {bm25_rank:<9}  {rrf}"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_eval(
    report: EvalReport,
    generated: AnswerEvalReport | None = None,
) -> str:
    """Format extractive scores, then generated scores when ``generated`` is set.

    The per-query lines stay extractive. Generated aggregates are four separate
    lines and are not folded into ``answer_accuracy``.
    """
    lines = [
        f"Eval queries={len(report.results)} k={report.k}",
        f"recall_at_k={report.mean_recall_at_k:.3f}",
        f"extractive_answer_accuracy={report.answer_accuracy:.3f}",
    ]
    if generated is not None:
        lines.extend(
            [
                f"generated_key_fact={generated.generated_key_fact:.3f}",
                f"citation_complete={generated.citation_complete:.3f}",
                f"groundedness={generated.groundedness:.3f}",
                f"conflict_handling={generated.conflict_handling:.3f}",
            ]
        )
    lines.append("")
    for result in report.results:
        lines.append(
            f"{result.query_id}  recall={result.recall_at_k:.0f}  "
            f"accurate={int(result.answer_correct)}  {result.retrieval_label.value}"
        )
        lines.append(
            f"  {result.answer.doc_name}  v{result.answer.version}  "
            f"{result.answer.status}  {result.answer.section}"
        )
        if result.retrieval_label is RetrievalLabel.DATA_QUALITY_FIXTURE:
            lines.append(f"  {result.note}")
    return "\n".join(lines).rstrip() + "\n"


def main(
    argv: Sequence[str] | None = None,
    *,
    opener: Callable[[], Pipeline] | None = None,
    generator_factory: Callable[[], TextGenerator] | None = None,
) -> int:
    """Run ``query``, ``compare``, ``answer``, or ``eval``. Return a process status code.

    ``opener`` defaults to :func:`open_default_pipeline`. Tests pass a
    pipeline built on a temporary Chroma directory and a fake scorer.
    ``generator_factory`` defaults to :func:`generator_from_env`. ``answer``
    and ``eval`` both use it. ``query`` and ``compare`` do not. ``compare``
    prints dense and RRF rows and does not call the cross-encoder.

    Raises:
        Nothing. Pipeline failures are printed to stderr and returned as 1.
        Argument errors are returned as 2.
    """
    parser = _parser()
    try:
        args = parser.parse_args(None if argv is None else list(argv))
    except SystemExit as exc:
        code = exc.code
        return 2 if code is None else int(code)

    pipeline: Pipeline | None = None
    try:
        pipeline = (opener or open_default_pipeline)()
        command = str(args.command)
        if command == "query":
            question = " ".join(str(part) for part in args.question)
            k = int(args.k)
            hits = pipeline.retriever.search(question, k=k)
            text = render_query(
                question,
                hits,
                documents=len(pipeline.documents),
                chunk_count=pipeline.chunk_count,
            )
        elif command == "compare":
            question = " ".join(str(part) for part in args.question)
            k = int(args.k)
            dense_hits = pipeline.index.search(question, k=k)
            hybrid_hits = pipeline.retriever.hybrid.search(question, k=k)
            text = render_compare(question, dense_hits, hybrid_hits, k=k)
        elif command == "answer":
            question = " ".join(str(part) for part in args.question)
            k = int(args.k)
            hits = pipeline.retriever.search(question, k=k)
            factory = generator_factory or generator_from_env
            text = render_answer(generate_answer(question, hits, factory()))
        elif command == "eval":
            documents = pipeline.documents
            factory = generator_factory or generator_from_env
            extractive = evaluate_retrieval(
                pipeline.retriever,
                bind_gold(evaluation_cases(), documents),
                k=EVAL_K,
            )
            generated = evaluate_answers(
                pipeline.retriever,
                bind_answer_gold(answer_cases(), documents),
                factory(),
                k=EVAL_K,
            )
            text = render_eval(extractive, generated)
        else:
            sys.stderr.write(f"error: unknown command {command}\n")
            return 2
        sys.stdout.write(text)
        return 0
    except RagLabError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    finally:
        if pipeline is not None:
            pipeline.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag_lab",
        description=(
            "Retrieve policy passages, compare dense and RRF ranks, generate "
            "a grounded answer, or print Recall@K and answer accuracy."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    query = sub.add_parser(
        "query",
        help="Retrieve chunks and print the extractive cited answer",
    )
    query.add_argument("question", nargs="+", help="Question text")
    query.add_argument(
        "--k",
        type=int,
        default=RETRIEVE_K,
        help=f"Chunks to return (default: {RETRIEVE_K})",
    )
    compare = sub.add_parser(
        "compare",
        help="Print dense-only and RRF hits side by side",
    )
    compare.add_argument("question", nargs="+", help="Question text")
    compare.add_argument(
        "--k",
        type=int,
        default=RETRIEVE_K,
        help=f"Ranks to print (default: {RETRIEVE_K})",
    )
    answer = sub.add_parser(
        "answer",
        help="Retrieve chunks and generate a grounded answer with Sources",
    )
    answer.add_argument("question", nargs="+", help="Question text")
    answer.add_argument(
        "--k",
        type=int,
        default=RETRIEVE_K,
        help=f"Chunks to pass to the generator (default: {RETRIEVE_K})",
    )
    sub.add_parser(
        "eval",
        help=(
            f"Print Recall@{EVAL_K}, extractive accuracy, and generated-answer "
            "scores for the production questions"
        ),
    )
    return parser


def _citation(hit: SearchHit | HybridHit | RerankedHit) -> dict[str, str]:
    values: dict[str, str] = {}
    for field in _CITATION_FIELDS:
        raw = hit.metadata.get(field, "")
        if not isinstance(raw, str) or not raw.strip():
            raise EvalError(f"Chunk {hit.chunk_id} is missing {field}")
        values[field] = raw
    return values


def _cite_label(hit: SearchHit | HybridHit | RerankedHit) -> str:
    meta = _citation(hit)
    return f"{meta['doc_name']}, {meta['section']}, {meta['version']}, {meta['status']}"


def _rank_cell(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines()) or "  "
