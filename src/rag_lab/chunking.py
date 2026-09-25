"""Static recursive hierarchical chunking with structural types.

Splitters are a fixed coarse-to-fine ladder (not an LLM). A unit that still
exceeds ``CHUNK_SIZE`` is recursed into the next finer type. Sibling
headings are never merged, so generated answers can cite a real section name. Adjacent
paragraphs and sentences are packed greedily up to the size budget. Character
windows with overlap are the last resort.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from rag_lab.config import CHUNK_OVERLAP, CHUNK_SIZE
from rag_lab.corpus import Document
from rag_lab.exceptions import ChunkingError

_HEADING_H2 = re.compile(r"(?m)(?=^## )")
_HEADING_H3 = re.compile(r"(?m)(?=^### )")
_HEADING_TITLE = re.compile(r"^#{2,3}\s+(.+)$", re.MULTILINE)
_HEADING_ONLY = re.compile(r"^#{1,6}\s+\S.*$")
_LIST_ITEM = re.compile(r"(?m)(?=^[-*] )")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z#\"“])")
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_PREAMBLE_SECTION = "Preamble"


class ChunkType(StrEnum):
    """Structural type of a chunk in the static hierarchy."""

    DOCUMENT = "document"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    LIST = "list"
    SENTENCE = "sentence"
    WINDOW = "window"


@dataclass(frozen=True)
class Chunk:
    """A typed text window plus attribution metadata for later retrieval."""

    chunk_id: str
    text: str
    chunk_type: ChunkType
    metadata: dict[str, str] = field(default_factory=dict)


def _validate_window(*, size: int, overlap: int) -> None:
    if size <= 0 or not 0 <= overlap < size:
        raise ChunkingError(f"Invalid size={size}, overlap={overlap}")


def _slug(value: str) -> str:
    slug = _SLUG_STRIP.sub("-", value.lower()).strip("-")
    return slug or "section"


def _heading_title(text: str) -> str | None:
    match = _HEADING_TITLE.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def _is_heading_only(text: str) -> bool:
    return bool(_HEADING_ONLY.fullmatch(text.strip()))


def _split_parts(text: str, chunk_type: ChunkType) -> list[str]:
    if chunk_type is ChunkType.SECTION:
        parts = _HEADING_H2.split(text)
    elif chunk_type is ChunkType.SUBSECTION:
        parts = _HEADING_H3.split(text)
    elif chunk_type is ChunkType.PARAGRAPH:
        parts = re.split(r"\n\s*\n", text)
    elif chunk_type is ChunkType.LIST:
        parts = _LIST_ITEM.split(text)
    elif chunk_type is ChunkType.SENTENCE:
        parts = _SENTENCE.split(text)
    else:
        return [text]
    return [part.strip() for part in parts if part.strip() and not _is_heading_only(part)]


def _join_parts(parts: list[str], chunk_type: ChunkType) -> str:
    if chunk_type is ChunkType.SENTENCE:
        return " ".join(parts)
    if chunk_type is ChunkType.LIST:
        return "\n".join(parts)
    return "\n\n".join(parts)


def _window_split(text: str, *, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    step = size - overlap
    windows: list[str] = []
    start = 0
    while start < len(text):
        windows.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += step
    return windows


_HIERARCHY: tuple[ChunkType, ...] = (
    ChunkType.SECTION,
    ChunkType.SUBSECTION,
    ChunkType.PARAGRAPH,
    ChunkType.LIST,
    ChunkType.SENTENCE,
    ChunkType.WINDOW,
)

_HEADING_TYPES = frozenset({ChunkType.SECTION, ChunkType.SUBSECTION})
_PACK_TYPES = frozenset({ChunkType.PARAGRAPH, ChunkType.LIST, ChunkType.SENTENCE})


def _with_heading(text: str, heading_stack: tuple[str, ...]) -> str:
    if not heading_stack or text.startswith("#"):
        return text
    prefix = "\n".join(f"## {title}" for title in heading_stack)
    return f"{prefix}\n\n{text}"


def _leaf(
    text: str,
    chunk_type: ChunkType,
    heading_stack: tuple[str, ...],
    *,
    size: int,
    overlap: int,
) -> list[tuple[str, ChunkType, tuple[str, ...]]]:
    titled = _with_heading(text, heading_stack)
    if _is_heading_only(titled):
        return []
    if len(titled) <= size:
        return [(titled, chunk_type, heading_stack)]
    return [
        (window, ChunkType.WINDOW, heading_stack)
        for window in _window_split(titled, size=size, overlap=overlap)
    ]


def _split_recursive(
    text: str,
    *,
    size: int,
    overlap: int,
    level: int,
    parent_type: ChunkType,
    heading_stack: tuple[str, ...],
) -> list[tuple[str, ChunkType, tuple[str, ...]]]:
    stripped = text.strip()
    if not stripped:
        return []

    if level >= len(_HIERARCHY) - 1:
        return _leaf(stripped, ChunkType.WINDOW, heading_stack, size=size, overlap=overlap)

    chunk_type = _HIERARCHY[level]
    # Headings always split so sibling sections keep distinct types and titles,
    # even when the combined document would fit in one 500-char window.
    if chunk_type not in _HEADING_TYPES and len(stripped) <= size:
        return _leaf(stripped, parent_type, heading_stack, size=size, overlap=overlap)

    parts = _split_parts(stripped, chunk_type)
    if len(parts) <= 1:
        if len(stripped) <= size:
            return _leaf(stripped, parent_type, heading_stack, size=size, overlap=overlap)
        return _split_recursive(
            stripped,
            size=size,
            overlap=overlap,
            level=level + 1,
            parent_type=chunk_type,
            heading_stack=heading_stack,
        )

    pack = chunk_type in _PACK_TYPES
    emitted: list[tuple[str, ChunkType, tuple[str, ...]]] = []
    buffer: list[str] = []

    def recurse(piece: str, stack: tuple[str, ...]) -> None:
        emitted.extend(
            _split_recursive(
                piece,
                size=size,
                overlap=overlap,
                level=level + 1,
                parent_type=chunk_type,
                heading_stack=stack,
            )
        )

    def flush() -> None:
        if not buffer:
            return
        joined = _join_parts(buffer, chunk_type)
        buffer.clear()
        if len(joined) <= size:
            emitted.extend(_leaf(joined, chunk_type, heading_stack, size=size, overlap=overlap))
            return
        recurse(joined, heading_stack)

    for part in parts:
        title = _heading_title(part) if chunk_type in _HEADING_TYPES else None
        child_stack = heading_stack + ((title,) if title else ())

        if len(part) > size:
            flush()
            recurse(part, child_stack or heading_stack)
            continue

        if not pack:
            flush()
            emitted.extend(
                _leaf(
                    part,
                    chunk_type,
                    child_stack or heading_stack,
                    size=size,
                    overlap=overlap,
                )
            )
            continue

        candidate = _join_parts([*buffer, part], chunk_type)
        if buffer and len(candidate) > size:
            flush()
        buffer.append(part)

    flush()
    return emitted


def chunk_text(
    text: str,
    *,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[str, ChunkType, tuple[str, ...]]]:
    """Split raw markdown using the static recursive hierarchy.

    Raises:
        ChunkingError: If ``size`` / ``overlap`` are invalid or ``text`` is empty.
    """
    _validate_window(size=size, overlap=overlap)
    if not text.strip():
        raise ChunkingError("Cannot chunk empty text")
    return _split_recursive(
        text,
        size=size,
        overlap=overlap,
        level=0,
        parent_type=ChunkType.DOCUMENT,
        heading_stack=(),
    )


def chunk_document(
    document: Document,
    *,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk one loaded policy and stamp attribution metadata on every piece.

    Raises:
        ChunkingError: If window parameters are invalid or the body is empty.
    """
    pieces = chunk_text(document.body, size=size, overlap=overlap)
    chunks: list[Chunk] = []
    source_file = document.path.name
    for index, (text, chunk_type, headings) in enumerate(pieces):
        section = headings[-1] if headings else _PREAMBLE_SECTION
        metadata = {
            **document.metadata,
            "section": section,
            "source_file": source_file,
            "chunk_index": str(index),
            "chunk_type": str(chunk_type),
        }
        chunk_id = f"{source_file}::{_slug(section)}::{index}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                chunk_type=chunk_type,
                metadata=metadata,
            )
        )
    return chunks


def chunk_corpus(
    documents: list[Document],
    *,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk every document in load order."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, size=size, overlap=overlap))
    if not chunks:
        raise ChunkingError("Corpus produced no chunks")
    return chunks
