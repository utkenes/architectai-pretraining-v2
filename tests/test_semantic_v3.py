"""Semantic linking v3 invariants and deterministic reporting tests."""

from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.semantic import (
    CANONICAL_CONCEPTS,
    annotate_document,
    coverage_report,
    group_adjacent_sections,
    normalize_concept,
)
from architectai_pretraining.sources import SourceConfig, matches_patterns
from architectai_pretraining.tokenizer import MockTokenCounter


def _doc(identifier: str, source: str, path: str, text: str, index: int) -> CorpusDocument:
    return CorpusDocument(
        id=identifier,
        source_id=source,
        category="software_architecture",
        category_hint="software_architecture",
        relative_path=path,
        text=text,
        section_title=f"section {index}",
        section_headings=[f"section {index}"],
        token_count=len(text.split()),
        metadata={"section_index": index},
    )


def test_category_hint_is_not_final_category() -> None:
    doc = annotate_document(
        _doc("raft", "primer", "raft.md", "Raft uses consensus, quorum, and leader election.", 0)
    )
    assert doc.category_hint == "software_architecture"
    assert doc.primary_category == "distributed_systems"
    assert doc.category == doc.primary_category


def test_zero_signal_uses_auditable_category_hint_fallback_or_stays_unresolved() -> None:
    fallback = annotate_document(
        _doc("fallback", "source", "a.md", "This prose has no taxonomy signals.", 0)
    )
    assert fallback.primary_category == "software_architecture"
    assert fallback.category_confidence == 0.1
    assert "fallback:category_hint" in fallback.category_evidence
    unresolved = annotate_document(
        CorpusDocument(id="unknown", source_id="source", category="legacy_unknown", text="Plain prose.")
    )
    assert unresolved.primary_category is None
    assert unresolved.metadata["classification_unresolved"] is True


def test_concept_normalization_is_safe_and_canonical() -> None:
    assert normalize_concept("fault tolerance") == "fault-tolerance"
    assert normalize_concept("fault_tolerance") == "fault-tolerance"
    assert normalize_concept("fault-tolerance") == "fault-tolerance"
    assert normalize_concept("leader election") == "leader-election"
    assert normalize_concept("unrelated thing") is None


def test_candidates_do_not_mutate_canonical_vocabulary() -> None:
    doc = annotate_document(
        _doc("clock", "s", "a.md", "Logical clocks order distributed events.", 0)
    )
    assert "logical-clocks" in doc.candidate_concepts
    assert "logical-clocks" not in CANONICAL_CONCEPTS


def test_candidate_audit_includes_counts_and_source_qualified_provenance() -> None:
    first = annotate_document(_doc("one", "one", "notes.md", "Logical clocks order events.", 0))
    second = annotate_document(_doc("two", "two", "notes.md", "Logical clocks order messages.", 0))
    report = coverage_report([first, second], min_tokens=1)
    candidate = report["candidate_concepts"]["logical-clocks"]
    assert candidate["source_count"] == 2
    assert candidate["document_count"] == 2
    assert candidate["documents"] == ["one:notes.md", "two:notes.md"]


def test_adjacent_related_sections_group_in_order_without_cross_document_text() -> None:
    counter = MockTokenCounter()
    raft = annotate_document(_doc("one", "s", "raft.md", "Raft consensus and leader election.", 0))
    replication = annotate_document(
        _doc("two", "s", "raft.md", "Raft log replication needs quorum.", 1)
    )
    unrelated = annotate_document(_doc("three", "s", "other.md", "API design uses versioning.", 0))
    grouped = group_adjacent_sections([replication, unrelated, raft], counter, 100)
    assert len(grouped) == 2
    joined = next(doc for doc in grouped if doc.relative_path == "raft.md")
    assert joined.text.index("leader election") < joined.text.index("log replication")
    assert "API design" not in joined.text
    assert joined.section_headings == ["section 0", "section 1"]


def test_grouping_respects_token_budget() -> None:
    counter = MockTokenCounter()
    first = annotate_document(_doc("one", "s", "a.md", "consensus " * 20, 0))
    second = annotate_document(_doc("two", "s", "a.md", "replication " * 20, 1))
    assert len(group_adjacent_sections([first, second], counter, 25)) == 2


def test_coverage_counts_diversity_and_dominance() -> None:
    a = annotate_document(_doc("a", "one", "a.md", "Raft consensus uses a quorum.", 0))
    b = annotate_document(_doc("b", "one", "b.md", "Raft consensus coordinates replicas.", 0))
    c = annotate_document(_doc("c", "two", "c.md", "Consensus is useful for consistency.", 0))
    report = coverage_report(
        [a, b, c], min_tokens=1, min_sources=2, min_documents=2, max_dominant_source_share=0.60
    )
    consensus = report["concept_coverage"]["consensus"]
    assert consensus["sources"] == 2
    assert consensus["documents"] == 3
    assert consensus["training_units"] == 3
    assert consensus["status"] == ["HIGH_SOURCE_CONCENTRATION"]
    assert report["category_coverage"]["distributed_systems"]["sources"] == 2


def test_new_source_policies_exclude_requested_paths(tmp_path: object) -> None:
    root = tmp_path / "repo"  # type: ignore[operator]
    (root / "exams").mkdir(parents=True)
    (root / "extra").mkdir()
    lecture = root / "l06-raft.md"
    lecture.write_text("Raft", encoding="utf-8")
    (root / "exams" / "exam.md").write_text("no", encoding="utf-8")
    pbft = root / "extra" / "pbft.md"
    pbft.write_text("PBFT", encoding="utf-8")
    config = SourceConfig(
        id="mit",
        name="MIT",
        category="distributed_systems",
        include_patterns=["l*.md", "extra/pbft.md"],
        exclude_patterns=["exams/**"],
    )
    assert matches_patterns(lecture, root, config.include_patterns, config.exclude_patterns)
    assert matches_patterns(pbft, root, config.include_patterns, config.exclude_patterns)
    assert not matches_patterns(
        root / "exams" / "exam.md", root, config.include_patterns, config.exclude_patterns
    )
