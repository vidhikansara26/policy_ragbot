"""Day-one proof: embed two known texts, store both, retrieve the right one.

Suggested Approach steps 2-3. Every stage prints to the terminal: the source
texts with their attribution metadata, their MiniLM vectors, the Chroma
collection the vectors land in, the metadata read back out of the store, and the
ranked cosine distances for two opposing queries. Runs the production embedder
and store against a throwaway Chroma directory, so the repo index is untouched.
This loop must pass before chunking, hybrid search, or reranking are trusted.
"""

from __future__ import annotations

import math
import tempfile
import textwrap
from pathlib import Path

from rag_lab.chunking import Chunk, ChunkType
from rag_lab.config import EMBED_DIM, EMBED_MODEL, EMBED_NORMALIZE
from rag_lab.embeddings import MiniLMEmbedder
from rag_lab.index import PolicyIndex

_RULE = "=" * 78
_WIDTH = 76
_GUTTER = " " * 15

TEXT_A = (
    "Full-time employees are entitled to 15 days of Privilege Leave (PTO) per "
    "calendar year. Unused Privilege Leave lapses on 31 December and cannot be "
    "encashed or carried forward."
)
TEXT_B = (
    "The Environment, Health and Safety policy targets Net zero by 2040 across "
    "all Coforge delivery centres and offices, with water positive operations "
    "by the same year."
)

_CHUNKS: tuple[Chunk, ...] = (
    Chunk(
        chunk_id="known-text-a",
        text=TEXT_A,
        chunk_type=ChunkType.PARAGRAPH,
        metadata={
            "doc_name": "Human Rights Policy",
            "section": "3. Fair Wages and Remuneration",
            "version": "1.0",
            "status": "legacy",
        },
    ),
    Chunk(
        chunk_id="known-text-b",
        text=TEXT_B,
        chunk_type=ChunkType.PARAGRAPH,
        metadata={
            "doc_name": "Environment, Health and Safety Policy",
            "section": "2. Environmental Commitments",
            "version": "2024",
            "status": "current",
        },
    ),
)

# One query per stored text. Answering both proves the winner follows the
# query's semantics and is not a bias toward whichever text was stored first.
_QUERIES: tuple[tuple[str, str], ...] = (
    ("How many Privilege Leave / PTO days do I get?", "known-text-a"),
    ("When is the net zero emissions target?", "known-text-b"),
)


def _wrap(text: str, indent: str = " " * 6) -> str:
    """Fold long policy text into an indented block that fits a screenshot."""
    return textwrap.fill(text, width=_WIDTH, initial_indent=indent, subsequent_indent=indent)


def _preview(vector: list[float], count: int = 6) -> str:
    """Render the leading dimensions of a vector."""
    head = ", ".join(f"{value:+.4f}" for value in vector[:count])
    return f"[{head}, ... ]"


def _l2_norm(vector: list[float]) -> float:
    """Return the Euclidean length, which is 1.0 for a normalized vector."""
    return math.sqrt(sum(value * value for value in vector))


def _heading(step: str, title: str) -> None:
    print()
    print(f"{step} {title}")
    print("-" * 78)


def _show_texts() -> None:
    _heading("[1/4]", "Two known pieces of text")
    for chunk in _CHUNKS:
        meta = chunk.metadata
        print(f"  {chunk.chunk_id}   {meta['doc_name']} / {meta['section']}")
        print(f"{_GUTTER}  v{meta['version']}, status={meta['status']}, {len(chunk.text)} chars")
        print(_wrap(chunk.text))


def _show_embeddings(embedder: MiniLMEmbedder) -> None:
    """Encode both texts and print the vectors the store will receive.

    upsert() encodes again inside the index; this pass exists so the terminal
    shows the vectors rather than only their downstream effect.
    """
    _heading("[2/4]", "Embedding with sentence-transformers")
    print(f"  model        {EMBED_MODEL}")
    print(f"  dimension    {EMBED_DIM}")
    print(f"  normalize    {EMBED_NORMALIZE}  (cosine == inner product on unit vectors)")
    vectors = embedder.embed_documents([chunk.text for chunk in _CHUNKS])
    for chunk, vector in zip(_CHUNKS, vectors, strict=True):
        print(f"  {chunk.chunk_id}   dim={len(vector)}  L2 norm={_l2_norm(vector):.6f}")
        print(f"{_GUTTER}{_preview(vector)}")


def _show_store(index: PolicyIndex) -> None:
    _heading("[3/4]", "Storing both vectors in ChromaDB")
    index.upsert(list(_CHUNKS))
    print("  backend      chromadb.PersistentClient")
    print(f"  path         {index.persist_directory}")
    print(f"  collection   {index.collection_name}")
    print(f"  space        {index.collection_metadata['hnsw:space']}  (lower is closer)")
    print(f"  stored       {index.count()} vectors")
    for chunk in _CHUNKS:
        stored = index.get_chunk(chunk.chunk_id)
        print(f"  read back    {stored.chunk_id} -> {stored.metadata}")


def _run_query(index: PolicyIndex, question: str, expected: str) -> bool:
    """Query the store, print every hit with its distance, and check rank 1."""
    print(f"  query        {question!r}")
    hits = index.search(question, k=len(_CHUNKS))
    if not hits:
        print("  FAIL         the store returned no hits")
        return False
    for rank, hit in enumerate(hits, start=1):
        marker = "   <-- closest" if rank == 1 else ""
        print(
            f"  rank {rank}       distance={hit.distance:.4f}  "
            f"cosine={1.0 - hit.distance:+.4f}  {hit.chunk_id}{marker}"
        )
        print(
            f"{_GUTTER}{hit.metadata['doc_name']} / {hit.metadata['section']} "
            f"(v{hit.metadata['version']}, {hit.metadata['status']})"
        )
        print(_wrap(hit.text, indent=_GUTTER))
    if len(hits) > 1:
        margin = hits[1].distance - hits[0].distance
        print(f"  margin       {margin:.4f} cosine distance between rank 1 and rank 2")
    if hits[0].chunk_id != expected:
        print(f"  FAIL         expected {expected} at rank 1, got {hits[0].chunk_id}")
        return False
    print(f"  PASS         {expected} is closest, as expected")
    return True


def main() -> int:
    """Print the whole embed, store, and retrieve loop on two known texts."""
    print(_RULE)
    print(" MINIMAL EMBED -> STORE -> RETRIEVE LOOP    (Suggested Approach steps 2-3)")
    print(_RULE)

    with tempfile.TemporaryDirectory() as tmp:
        embedder = MiniLMEmbedder()
        index = PolicyIndex(Path(tmp) / "chroma", embedder=embedder)
        try:
            _show_texts()
            _show_embeddings(embedder)
            _show_store(index)

            _heading("[4/4]", "Retrieving: one query per stored text")
            outcomes: list[bool] = []
            for position, (question, expected) in enumerate(_QUERIES, start=1):
                print(f"  --- query {position} of {len(_QUERIES)} ---")
                outcomes.append(_run_query(index, question, expected))
                print()

            print(_RULE)
            if all(outcomes):
                print(" RESULT  PASS - both queries returned the more relevant of the two texts.")
                print("         embed -> store -> retrieve is proven on known input.")
                print("         Safe to build chunking, hybrid search, and reranking on top.")
                print(_RULE)
                return 0
            failures = outcomes.count(False)
            print(f" RESULT  FAIL - {failures} of {len(outcomes)} queries returned the wrong text.")
            print(_RULE)
            return 1
        finally:
            index.close()


if __name__ == "__main__":
    raise SystemExit(main())