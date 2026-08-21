"""Semantic v3 invariants: taxonomy, grouping, coverage, and source policies."""

from architectai_pretraining.corpus_v2 import load_corpus_v2_config
from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.semantic import (
    annotate_document,
    coverage_report,
    group_adjacent_sections,
    normalize_concept,
)
from architectai_pretraining.tokenizer import MockTokenCounter


def _doc(identifier: str, source: str, path: str, text: str, index: int = 0) -> CorpusDocument:
    return annotate_document(CorpusDocument(
        id=identifier,
        source_id=source,
        category="domain_driven_design",
        category_hint="domain_driven_design",
        relative_path=path,
        section_title=f"Section {index}",
        section_headings=[f"Section {index}"],
        text=text,
        token_count=len(text.split()),
        metadata={"section_index": index, "source_document_id": f"{source}:{path}"},
    ))


def test_category_hint_is_a_weak_prior_and_v3_unit_has_one_primary_category() -> None:
    doc = _doc("raft", "ddd-source", "notes.md", "Raft uses consensus, quorum, and leader election.")
    assert doc.primary_category == "distributed_systems"
    assert doc.category == doc.primary_category
    assert doc.schema_version == 3


def test_concept_normalization_and_candidates_are_controlled() -> None:
    assert normalize_concept("fault tolerance") == "fault-tolerance"
    assert normalize_concept("fault_tolerance") == "fault-tolerance"
    assert normalize_concept("fault-tolerance") == "fault-tolerance"
    assert normalize_concept("lease-based leadership") is None
    assert "leader-election" in _doc("leader", "s", "leader.md", "A leader_election makes the service fault-tolerant.").related_concepts
    doc = _doc("candidate", "s", "notes.md", "Lease-based leadership relies on anti-entropy.")
    assert doc.candidate_concepts == ["anti-entropy", "lease-based-leadership"]
    assert "anti-entropy" not in doc.related_concepts


def test_grouping_preserves_order_budget_and_document_boundary() -> None:
    first = _doc("one", "s", "raft.md", "Raft leader election uses quorum.", 0)
    second = _doc("two", "s", "raft.md", "Log replication requires quorum.", 1)
    unrelated = _doc("three", "s", "raft.md", "A bounded context contains an aggregate.", 2)
    other_document = _doc("four", "s", "other.md", "Raft leader election uses quorum.", 0)
    grouped = group_adjacent_sections([other_document, unrelated, second, first], MockTokenCounter(), 20)
    assert len(grouped) == 3
    combined = next(doc for doc in grouped if doc.relative_path == "raft.md" and "one" in doc.metadata.get("grouped_section_ids", []))
    assert combined.text.index("leader election") < combined.text.index("Log replication")
    assert combined.metadata["grouped_section_ids"] == ["one", "two"]
    assert all(doc.source_id == "s" for doc in grouped)


def test_coverage_counts_diversity_and_concentration() -> None:
    left = _doc("left", "one", "a.md", "Raft consensus uses a quorum.")
    right = _doc("right", "two", "b.md", "Consensus and quorum improve availability.")
    report = coverage_report([left, right], min_tokens=1, min_sources=2, min_documents=2)
    assert report["concept_coverage"]["consensus"]["status"] == "healthy"
    assert report["concept_coverage"]["replication"]["status"] == ["NO_COVERAGE"]
    assert set(report["source_concept_matrix"]["consensus"]) == {"one", "two"}


def test_new_source_policies_are_selective_and_migration_friendly() -> None:
    config = load_corpus_v2_config("configs/corpus_v2.yaml")
    sources = {source.id: source for source in config.source_configs}
    assert len(sources) == 33
    assert sources["mit_6824_lecture_notes"].category_hint == "distributed_systems"
    assert sources["mit_6824_lecture_notes"].include_patterns == ["l*.md", "extra/pbft.md"]
    assert "**/ARCHITECTURE.md" in sources["mozilla_application_services"].include_patterns
    assert sources["architecture_center"].include_patterns != ["docs/**/*.md"]
    # Existing category entries remain valid aliases for migration.
    assert sources["nats_docs"].category_hint == "messaging_event_driven"
    assert sources["debezium_outbox_docs"].include_patterns == [
        "documentation/modules/ROOT/pages/transformations/outbox-event-router.adoc",
        "documentation/modules/ROOT/pages/transformations/mongodb-outbox-event-router.adoc",
    ]
    assert sources["dotnet_ddd_domain_events_docs"].source_token_cap == 25000
    assert sources["eventuate_tram_sagas_docs"].license_evidence_path == "LICENSE.md"
