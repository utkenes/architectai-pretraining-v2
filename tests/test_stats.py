"""Tests for CorpusStats and SourceStats reporting."""

from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.stats import calculate_stats


def test_calculate_stats_per_source() -> None:
    doc1 = CorpusDocument(
        id="doc-1",
        source_id="system_design_primer",
        category="architecture_patterns",
        title="System Design",
        text="Scalability patterns include load balancing and caching.",
    )
    doc2 = CorpusDocument(
        id="doc-2",
        source_id="etcd_docs",
        category="distributed_systems",
        title="Raft Consensus",
        text="etcd implements Raft consensus for state replication.",
    )

    source_names = {
        "system_design_primer": "The System Design Primer",
        "etcd_docs": "etcd Distributed Systems Documentation",
    }
    per_source_ingested = {"system_design_primer": 2, "etcd_docs": 1}
    per_source_rejected = {"system_design_primer": 1, "etcd_docs": 0}
    per_source_duplicates = {"system_design_primer": 0, "etcd_docs": 0}

    stats = calculate_stats(
        train_docs=[doc1],
        val_docs=[doc2],
        input_count=3,
        rejected_count=1,
        duplicates_removed=0,
        per_source_ingested=per_source_ingested,
        per_source_rejected=per_source_rejected,
        per_source_duplicates=per_source_duplicates,
        source_names=source_names,
    )

    assert stats.accepted_documents == 2
    assert stats.rejected_documents == 1
    assert "system_design_primer" in stats.source_stats
    assert stats.source_stats["system_design_primer"].documents_ingested == 2
    assert stats.source_stats["system_design_primer"].documents_rejected == 1

    report = stats.to_formatted_report()
    assert "Per-Source Statistics:" in report
    assert "The System Design Primer (system_design_primer)" in report
