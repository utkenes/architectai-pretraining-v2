"""Tests for quality gate filtering."""

from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.quality import QualityGate, QualityGateConfig


def test_quality_gate_accepts_valid_document() -> None:
    gate = QualityGate(QualityGateConfig(min_char_length=50, min_word_count=10))
    doc = CorpusDocument(
        id="doc-valid",
        source_id="manual_docs",
        category="distributed_systems",
        text=(
            "The CAP theorem states that a distributed data store can at most provide "
            "two of three guarantees: Consistency, Availability, and Partition tolerance."
        ),
    )

    res = gate.validate(doc)
    assert res.passed is True
    assert len(res.rejection_reasons) == 0


def test_quality_gate_rejects_short_document() -> None:
    gate = QualityGate(QualityGateConfig(min_char_length=100, min_word_count=20))
    doc = CorpusDocument(
        id="doc-short",
        source_id="manual_docs",
        category="reliability",
        text="Too short text.",
    )

    res = gate.validate(doc)
    assert res.passed is False
    assert any("below minimum threshold" in r for r in res.rejection_reasons)


def test_quality_gate_rejects_excessive_symbols() -> None:
    gate = QualityGate(QualityGateConfig(max_symbol_ratio=0.30))
    doc = CorpusDocument(
        id="doc-symbols",
        source_id="manual_docs",
        category="software_design",
        text="### !!! $$$ %%% ^^^ &&& *** ((( ))) ___ +++ === {{{ }}} ::: <<< >>> ???",
    )

    res = gate.validate(doc)
    assert res.passed is False
    assert any("Symbol ratio" in r for r in res.rejection_reasons)


def test_filter_documents_helper() -> None:
    gate = QualityGate(QualityGateConfig(min_char_length=50, min_word_count=5))
    doc1 = CorpusDocument(
        id="doc-1",
        source_id="src-1",
        category="cloud_architecture",
        text=(
            "Cloud-native applications utilize microservices architecture and "
            "automated container management."
        ),
    )
    doc2 = CorpusDocument(
        id="doc-2",
        source_id="src-1",
        category="cloud_architecture",
        text="Short.",
    )

    accepted, rejected = gate.filter_documents([doc1, doc2])
    assert len(accepted) == 1
    assert accepted[0].id == "doc-1"
    assert len(rejected) == 1
    assert rejected[0][0].id == "doc-2"
