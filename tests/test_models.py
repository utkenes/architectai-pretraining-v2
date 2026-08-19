"""Tests for CorpusDocument data model."""

import json

import pytest
from pydantic import ValidationError

from architectai_pretraining.models import CorpusDocument


def test_corpus_document_valid_creation() -> None:
    doc = CorpusDocument(
        id="doc-101",
        source_id="manual_local_docs",
        source_url="https://example.org/docs/arch",
        license_id="CC-BY-4.0",
        category="architecture_patterns",
        title="Modular Monolith Pattern",
        text="A modular monolith is a deployment topology...",
        language="en",
        metadata={"author": "Architect"},
    )
    assert doc.id == "doc-101"
    assert doc.source_id == "manual_local_docs"
    assert doc.license_id == "CC-BY-4.0"
    assert doc.language == "en"
    assert doc.metadata["author"] == "Architect"


def test_corpus_document_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        CorpusDocument(
            id="doc-102",
            source_id="src-1",
            category="architecture_patterns",
            text="   ",
        )


def test_corpus_document_json_roundtrip() -> None:
    doc = CorpusDocument(
        id="doc-103",
        source_id="src-1",
        source_url=None,
        license_id=None,
        category="distributed_systems",
        title="Raft Consensus",
        text="Raft is a consensus algorithm designed to be understandable.",
    )
    json_str = doc.to_json_str()
    parsed = json.loads(json_str)

    assert parsed["id"] == "doc-103"
    assert parsed["license_id"] is None

    restored = CorpusDocument.from_json_str(json_str)
    assert restored == doc


def test_corpus_document_no_instruction_fields() -> None:
    doc = CorpusDocument(
        id="doc-104",
        source_id="src-1",
        category="reliability",
        text="Circuit breakers prevent cascading failure.",
    )
    dump = doc.model_dump()
    for forbidden in ["user", "assistant", "question", "answer", "instruction", "prompt"]:
        assert forbidden not in dump
