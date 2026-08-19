"""Tests for deduplication module."""

from architectai_pretraining.dedup import ExactDeduplicator
from architectai_pretraining.models import CorpusDocument


def test_exact_deduplication_removes_identical_docs() -> None:
    deduper = ExactDeduplicator()

    doc1 = CorpusDocument(
        id="doc-1",
        source_id="src-1",
        category="architecture_patterns",
        text="Microservices enable independent deployment.",
    )
    doc2 = CorpusDocument(
        id="doc-2",
        source_id="src-2",
        category="architecture_patterns",
        text="Microservices enable independent deployment.",  # Identical text
    )
    doc3 = CorpusDocument(
        id="doc-3",
        source_id="src-1",
        category="architecture_patterns",
        text="Modular Monolith avoids network overhead.",
    )

    res = deduper.deduplicate([doc1, doc2, doc3])

    assert res.input_documents == 3
    assert res.duplicates_removed == 1
    assert res.remaining_documents == 2
    assert [d.id for d in res.deduplicated_documents] == ["doc-1", "doc-3"]


def test_exact_deduplication_whitespace_normalization() -> None:
    deduper = ExactDeduplicator(normalize_whitespace=True)

    doc1 = CorpusDocument(
        id="doc-1",
        source_id="src-1",
        category="reliability",
        text="SRE principles focus  on \n SLOs and SLA target metrics.",
    )
    doc2 = CorpusDocument(
        id="doc-2",
        source_id="src-1",
        category="reliability",
        text="SRE principles focus on SLOs and SLA target metrics.",
    )

    res = deduper.deduplicate([doc1, doc2])
    assert res.duplicates_removed == 1
    assert len(res.deduplicated_documents) == 1
