"""Hierarchical chunking: structural types, packing, and data-quality fixtures."""

import pytest

from rag_lab import config
from rag_lab.chunking import ChunkType, chunk_corpus, chunk_document, chunk_text
from rag_lab.corpus import Document, load_corpus
from rag_lab.exceptions import ChunkingError


def test_invalid_window_raises() -> None:
    with pytest.raises(ChunkingError, match="Invalid size"):
        chunk_text("hello world " * 20, size=100, overlap=100)


def test_empty_text_raises() -> None:
    with pytest.raises(ChunkingError, match="empty"):
        chunk_text("   ")


def test_short_document_stays_one_typed_leaf() -> None:
    pieces = chunk_text("# Title\n\nA short policy body.", size=500, overlap=100)
    assert len(pieces) == 1
    text, chunk_type, _headings = pieces[0]
    assert chunk_type is ChunkType.DOCUMENT
    assert "short policy" in text


def test_sibling_headings_are_not_merged() -> None:
    body = (
        "## Purpose\n\n"
        + ("Keep the board diverse. " * 8)
        + "\n\n## Vision\n\n"
        + ("Serve shareholders and community. " * 8)
    )
    pieces = chunk_text(body, size=500, overlap=100)
    types = [chunk_type for _, chunk_type, _ in pieces]
    assert ChunkType.SECTION in types
    titles = [" ".join(headings) for _, _, headings in pieces]
    assert any("Purpose" in title for title in titles)
    assert any("Vision" in title for title in titles)


def test_paragraphs_pack_until_size_budget() -> None:
    para_a = "Alpha clause. " * 12  # ~168 chars
    para_b = "Beta clause. " * 12
    filler = "Oversized section preamble so the parent exceeds the budget. " * 12
    body = f"## Fair Wages\n\n{filler}\n\n{para_a}\n\n{para_b}"
    pieces = chunk_text(body, size=500, overlap=100)
    packed = [piece for piece in pieces if "Alpha" in piece[0] and "Beta" in piece[0]]
    assert packed, "adjacent short paragraphs should pack into one chunk"
    text, chunk_type, _ = packed[0]
    assert chunk_type is ChunkType.PARAGRAPH
    assert len(text) <= 500


def test_window_overlap_is_applied_as_last_resort() -> None:
    blob = "abcdefghij" * 80  # 800 chars, no headings or sentence breaks
    pieces = chunk_text(blob, size=500, overlap=100)
    assert len(pieces) == 2
    first, second = pieces[0][0], pieces[1][0]
    assert pieces[0][1] is ChunkType.WINDOW
    assert first[-100:] == second[:100]


def test_corpus_chunks_respect_size_and_metadata() -> None:
    chunks = chunk_corpus(load_corpus())
    assert len(chunks) > len(load_corpus())
    for chunk in chunks:
        assert len(chunk.text) <= config.CHUNK_SIZE
        assert chunk.chunk_type in ChunkType
        assert chunk.metadata["doc_name"]
        assert chunk.metadata["version"]
        assert chunk.metadata["status"]
        assert "chunk_type" in chunk.metadata


def test_planted_pto_clause_stays_with_legacy_metadata() -> None:
    legacy = next(
        document
        for document in load_corpus()
        if document.doc_name == config.PLANTED_DOC_NAME
        and document.version == config.LEGACY_POLICY_VERSION
    )
    chunks = chunk_document(legacy)
    matching = [chunk for chunk in chunks if "Privilege Leave (PTO)" in chunk.text]
    assert matching
    assert any(f"{config.LEGACY_PTO_DAYS} days" in chunk.text for chunk in matching)
    assert all(chunk.metadata["status"] == "legacy" for chunk in matching)
    assert all(chunk.metadata["version"] == "1.0" for chunk in matching)


def test_heading_only_chunks_are_dropped() -> None:
    body = "## Definitions\n\n" + ("- **Term** is a long definition that fills the section. " * 20)
    pieces = chunk_text(body, size=500, overlap=100)
    assert pieces
    assert pieces
    assert all(piece[0].strip() != "## Definitions" for piece in pieces)
    assert all(len(piece[0].strip()) > 40 for piece in pieces)


def test_preamble_is_not_labeled_with_yaml_section() -> None:
    legacy = next(
        document
        for document in load_corpus()
        if document.version == config.LEGACY_POLICY_VERSION
        and document.doc_name == config.PLANTED_DOC_NAME
    )
    chunks = chunk_document(legacy)
    preamble = chunks[0]
    assert preamble.metadata["section"] == "Preamble"
    assert "LEGACY" in preamble.text or "Human Rights" in preamble.text


def test_bullet_items_are_not_cut_mid_line() -> None:
    bullets = "\n".join(
        f"- Item {index:02d} stays whole even when the list is long enough to exceed the budget."
        for index in range(12)
    )
    body = f"## Objectives\n\n{bullets}"
    pieces = chunk_text(body, size=500, overlap=100)
    for text, chunk_type, _headings in pieces:
        for line in text.splitlines():
            if line.startswith("- Item"):
                assert line.endswith("budget.")
        assert chunk_type in {ChunkType.LIST, ChunkType.SECTION, ChunkType.PARAGRAPH}


def test_long_list_items_are_not_packed_together() -> None:
    """Two self-contained bullets stay in separate chunks."""
    clause = "reduce the measured impact across every operating site and report it. " * 3
    bullets = "\n".join(f"- **Target {name} by 2040:** {clause}" for name in "ABC")
    body = f"## Objectives\n\n{bullets}"
    pieces = chunk_text(body, size=500, overlap=100)
    texts = [text for text, _, _ in pieces]
    assert any("Target A" in text and "Target B" not in text for text in texts)
    assert not any("Target A" in text and "Target B" in text for text in texts)


def test_short_list_items_still_pack_for_context() -> None:
    """A bullet below the atomic floor keeps its siblings so it is answerable."""
    bullets = "\n".join(
        f"- Duty {index:02d}: keep the workplace safe and report incidents." for index in range(12)
    )
    pieces = chunk_text(f"## Duties\n\n{bullets}", size=500, overlap=100)
    packed = [text for text, chunk_type, _ in pieces if chunk_type is ChunkType.LIST]
    assert packed
    assert any(text.count("- Duty") > 1 for text in packed)


def test_parallel_objectives_do_not_share_a_chunk() -> None:
    """The "by 2040" objectives are near-duplicates; the gold bullet is indexed alone."""
    gold = [chunk for chunk in chunk_corpus(load_corpus()) if "Net zero by 2040" in chunk.text]
    assert len(gold) == 1
    assert gold[0].chunk_type is ChunkType.LIST
    assert "Water positive by 2040" not in gold[0].text
    assert "Zero waste to landfill by 2040" not in gold[0].text


def test_h3_subsections_keep_separate_identities() -> None:
    body = (
        "## Parent\n\n"
        + ("Preamble for the parent section. " * 20)
        + "\n\n### Child A\n\n"
        + ("Alpha child content. " * 8)
        + "\n\n### Child B\n\n"
        + ("Beta child content. " * 8)
    )
    pieces = chunk_text(body, size=500, overlap=100)
    titles = [" | ".join(headings) for _, _, headings in pieces]
    assert any("Child A" in title for title in titles)
    assert any("Child B" in title for title in titles)


def test_chunk_ids_are_unique_across_corpus() -> None:
    chunks = chunk_corpus(load_corpus())
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    "needle,doc_name,version",
    [
        ("15 days", "Human Rights Policy", "1.0"),
        ("Privilege Leave (PTO)", "Human Rights Policy", "1.0"),
        ("hr.helpdesk@niit-tech.com", "Human Rights Policy", "1.0"),
        ("All.HR@coforge.com", "Human Rights Policy", "2.0"),
        ("shrc@coforge.com", "Policy Against Sexual Harassment at Workplace", "2024"),
        ("whistleblower@coforge.com", "Whistleblower Policy", "1.5"),
        ("Net zero by 2040", "Global Environment Health and Safety Policy", "5.0"),
        ("80%", "Modern Slavery Act Statement", "2024"),
    ],
)
def test_eval_facts_live_in_one_attributed_chunk(needle: str, doc_name: str, version: str) -> None:
    hits = [
        chunk
        for chunk in chunk_corpus(load_corpus())
        if needle in chunk.text
        and chunk.metadata["doc_name"] == doc_name
        and chunk.metadata["version"] == version
    ]
    assert hits, f"{needle!r} missing from {doc_name} v{version}"
    assert all(needle in chunk.text for chunk in hits)


def test_chunk_document_rejects_empty_body() -> None:
    document = Document(
        path=load_corpus()[0].path,
        body="   ",
        metadata={
            "doc_name": "Empty",
            "section": "None",
            "version": "0",
            "status": "current",
        },
    )
    with pytest.raises(ChunkingError):
        chunk_document(document)
