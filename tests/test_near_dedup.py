"""Unit tests for MinHash LSH near-duplicate detection."""

from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.near_dedup import MinHashLSHDeduplicator


def test_minhash_lsh_near_duplicate_detection() -> None:
    doc1 = CorpusDocument(
        id="doc_orig",
        title="Microservice Architecture Overview",
        text="""Microservice architecture structures an application as a collection of services.
Services are independently deployable and communicate over lightweight protocols such as HTTP REST or gRPC.
Event-driven communication ensures loose coupling between domain boundaries.""",
        source_id="src1",
        category="architecture_patterns",
        license_id="MIT",
        language="en",
    )

    doc2 = CorpusDocument(
        id="doc_copy",
        title="Microservice Architecture Overview Copy",
        text="""Microservice architecture structures an application as a collection of services.
Services are independently deployable and communicate over lightweight protocols such as HTTP REST or gRPC.
Event-driven communication ensures loose coupling between domain boundaries.""",
        source_id="src2",
        category="architecture_patterns",
        license_id="MIT",
        language="en",
    )

    doc3 = CorpusDocument(
        id="doc_unique",
        title="ClickHouse Storage Engine Internals",
        text="""ClickHouse uses a MergeTree storage engine family designed for high-performance analytical queries.
Data is partitioned by key and written in sorted column parts that are asynchronously merged in the background.""",
        source_id="src3",
        category="database_architecture",
        license_id="Apache-2.0",
        language="en",
    )

    dedup = MinHashLSHDeduplicator(similarity_threshold=0.80)
    res = dedup.deduplicate([doc1, doc2, doc3])

    assert len(res.canonical_documents) == 2
    assert len(res.removed_documents) == 1
    assert len(res.clusters) == 1
    assert res.clusters[0].canonical_document_id == "doc_orig" or res.clusters[0].canonical_document_id == "doc_copy"
