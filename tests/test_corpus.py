"""Corpus loader: versioned Coforge policies with a stale Human Rights v1 fixture."""

from pathlib import Path

import pytest

from rag_lab import config
from rag_lab.corpus import body_word_count, load_corpus
from rag_lab.exceptions import DataLoadError


def _write_policy(directory: Path, *, role: str, name: str = "sample.md") -> Path:
    path = directory / name
    path.write_text(
        (
            "---\n"
            'doc_name: "Sample Policy"\n'
            'section: "Scope"\n'
            'version: "1.0"\n'
            'status: "current"\n'
            f"role: {role!r}\n"
            "---\n\n"
            "# Sample\n\nA short policy body for loader tests.\n"
        ),
        encoding="utf-8",
    )
    return path


def test_corpus_has_at_least_eight_documents() -> None:
    documents = load_corpus()
    assert len(documents) >= 8


def test_every_document_has_attribution_metadata() -> None:
    for document in load_corpus():
        for field in config.REQUIRED_METADATA_FIELDS:
            assert document.metadata[field], f"{document.path} missing {field}"
        assert document.role in config.ALLOWED_CORPUS_ROLES


def test_planted_human_rights_version_conflict() -> None:
    human_rights = [
        document for document in load_corpus() if document.doc_name == config.PLANTED_DOC_NAME
    ]
    by_version = {document.version: document for document in human_rights}

    assert config.LEGACY_POLICY_VERSION in by_version
    assert config.CURRENT_POLICY_VERSION in by_version

    legacy = by_version[config.LEGACY_POLICY_VERSION]
    current = by_version[config.CURRENT_POLICY_VERSION]

    assert legacy.status == "legacy"
    assert legacy.role == config.CORPUS_ROLE_FIXTURE
    assert legacy.path.name == config.FIXTURE_SOURCE_FILE
    assert current.status == "current"
    assert current.role == config.CORPUS_ROLE_PRIMARY
    assert f"{config.LEGACY_PTO_DAYS} days" in legacy.body
    assert "Privilege Leave (PTO)" in legacy.body
    assert f"{config.LEGACY_PTO_DAYS} days" not in current.body
    assert config.LEGACY_HR_COMPLAINTS_EMAIL in legacy.body
    assert config.CURRENT_HR_COMPLAINTS_EMAIL in current.body
    assert config.CURRENT_HR_COMPLAINTS_EMAIL not in legacy.body


def test_current_policies_keep_source_urls() -> None:
    current_docs = [document for document in load_corpus() if document.status == "current"]
    assert current_docs
    for document in current_docs:
        source = document.metadata.get("source_url", "")
        assert source.startswith("https://investors.coforge.com/"), document.path


def test_four_primary_policies_are_in_assignment_word_band() -> None:
    """The assignment set is four current policies of 500–800 words."""
    by_name = {document.path.name: document for document in load_corpus()}
    assert set(by_name) >= config.PRIMARY_SOURCE_FILES
    for name in sorted(config.PRIMARY_SOURCE_FILES):
        document = by_name[name]
        words = body_word_count(document.body)
        assert document.role == config.CORPUS_ROLE_PRIMARY, name
        assert document.status == "current", name
        assert config.PRIMARY_WORD_MIN <= words <= config.PRIMARY_WORD_MAX, (
            f"{name} has {words} words; expected {config.PRIMARY_WORD_MIN}-"
            f"{config.PRIMARY_WORD_MAX}"
        )


def test_supplemental_policies_are_tagged() -> None:
    by_name = {document.path.name: document for document in load_corpus()}
    for name in sorted(config.SUPPLEMENTAL_SOURCE_FILES):
        document = by_name[name]
        assert document.role == config.CORPUS_ROLE_SUPPLEMENTAL, name
        assert document.status == "current", name


def test_corpus_roles_partition_the_files() -> None:
    documents = load_corpus()
    by_role = {
        config.CORPUS_ROLE_PRIMARY: set(),
        config.CORPUS_ROLE_SUPPLEMENTAL: set(),
        config.CORPUS_ROLE_FIXTURE: set(),
    }
    for document in documents:
        by_role[document.role].add(document.path.name)
    assert by_role[config.CORPUS_ROLE_PRIMARY] == config.PRIMARY_SOURCE_FILES
    assert by_role[config.CORPUS_ROLE_SUPPLEMENTAL] == config.SUPPLEMENTAL_SOURCE_FILES
    assert by_role[config.CORPUS_ROLE_FIXTURE] == {config.FIXTURE_SOURCE_FILE}


def test_legacy_fixture_keeps_fifteen_day_pto_and_is_not_length_gated() -> None:
    """Human Rights v1 stays a short stale fixture; length is not in the 500-800 band."""
    legacy = next(
        document for document in load_corpus() if document.path.name == config.FIXTURE_SOURCE_FILE
    )
    assert legacy.status == "legacy"
    assert legacy.role == config.CORPUS_ROLE_FIXTURE
    assert f"{config.LEGACY_PTO_DAYS} days of Privilege Leave (PTO)" in legacy.body
    words = body_word_count(legacy.body)
    assert words < config.PRIMARY_WORD_MIN


def test_invalid_role_is_rejected(tmp_path: Path) -> None:
    _write_policy(tmp_path, role="eval")
    with pytest.raises(DataLoadError, match="invalid role"):
        load_corpus(tmp_path)
