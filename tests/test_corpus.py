"""Part 1: Coforge policy corpus loads with required metadata and a planted v1 defect."""

from rag_lab import config
from rag_lab.corpus import load_corpus


def test_corpus_has_at_least_eight_documents() -> None:
    documents = load_corpus()
    assert len(documents) >= 8


def test_every_document_has_attribution_metadata() -> None:
    for document in load_corpus():
        for field in config.REQUIRED_METADATA_FIELDS:
            assert document.metadata[field], f"{document.path} missing {field}"


def test_planted_human_rights_version_conflict() -> None:
    human_rights = [
        document
        for document in load_corpus()
        if document.doc_name == config.PLANTED_DOC_NAME
    ]
    by_version = {document.version: document for document in human_rights}

    assert config.LEGACY_POLICY_VERSION in by_version
    assert config.CURRENT_POLICY_VERSION in by_version

    legacy = by_version[config.LEGACY_POLICY_VERSION]
    current = by_version[config.CURRENT_POLICY_VERSION]

    assert legacy.status == "legacy"
    assert current.status == "current"
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
