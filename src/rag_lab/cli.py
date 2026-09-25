"""Command-line runner for one query and for the eval scoreboard.

``query`` loads ``data/raw``, chunks, embeds, stores, fuses dense and BM25
with reciprocal rank fusion, reranks, and prints the returned chunks plus
the extractive answer. The answer is the rank-1 passage. It carries doc
name, section, and version. ``status=legacy`` is not filtered.

``eval`` runs the same pipeline over the production question set and prints
Recall@K and answer accuracy on separate lines. K is ``EVAL_K``. The
Privilege Leave row is labeled ``data_quality_fixture`` when Human Rights
Policy v1 is in the window.

No generator is called. Model ids, K, and the Chroma directory stay in
``rag_lab.config``.
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
from rag_lab.eval import EvalReport, RetrievalLabel, bind_gold, evaluate, evaluation_cases
from rag_lab.exceptions import DataLoadError, EvalError, RagLabError
from rag_lab.hybrid import HybridRetriever
from rag_lab.index import PolicyIndex
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
        raise EvalError(
            "Query returned no passages; cannot cite doc name, section, and version"
        )
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


def render_eval(report: EvalReport) -> str:
    """Format Recall@K and answer accuracy as separate lines, then each query."""
    lines = [
        f"Eval queries={len(report.results)} k={report.k}",
        f"recall_at_k={report.mean_recall_at_k:.3f}",
        f"answer_accuracy={report.answer_accuracy:.3f}",
        "",
    ]
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
) -> int:
    """Run ``query`` or ``eval``. Return a process status code.

    ``opener`` defaults to :func:`open_default_pipeline`. Tests pass a
    pipeline built on a temporary Chroma directory and a fake scorer.

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
        elif command == "eval":
            gold = bind_gold(evaluation_cases(), pipeline.documents)
            text = render_eval(evaluate(pipeline.retriever, gold, k=EVAL_K))
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
            "Run the policy pipeline for one question, or print Recall@K and "
            "answer accuracy for the eval set."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    query = sub.add_parser("query", help="Retrieve chunks and print the cited answer")
    query.add_argument("question", nargs="+", help="Question text")
    query.add_argument(
        "--k",
        type=int,
        default=RETRIEVE_K,
        help=f"Chunks to return (default: {RETRIEVE_K})",
    )
    sub.add_parser(
        "eval",
        help=f"Print Recall@{EVAL_K} and answer accuracy for the production questions",
    )
    return parser


def _citation(hit: RerankedHit) -> dict[str, str]:
    values: dict[str, str] = {}
    for field in _CITATION_FIELDS:
        raw = hit.metadata.get(field, "")
        if not isinstance(raw, str) or not raw.strip():
            raise EvalError(f"Chunk {hit.chunk_id} is missing {field}")
        values[field] = raw
    return values


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines()) or "  "
