"""Final sanitation and cleanliness regression tests using synthetic prose."""

from architectai_pretraining.cleaner import BoilerplateCleaner
from architectai_pretraining.corpus_v2 import (
    _prepare_for_sectioning,
    cleanliness_audit,
)
from architectai_pretraining.models import CorpusDocument


def _doc(text: str, *, title: str | None = None, section_title: str | None = None) -> CorpusDocument:
    return CorpusDocument(
        id="unit",
        source_id="source",
        category="distributed_systems",
        title=title,
        section_title=section_title,
        text=text,
    )


def test_flashcard_section_is_removed_but_rhetorical_question_and_design_checklist_remain() -> None:
    source = _doc(
        "# Leader Election\nRaft uses a quorum to preserve availability.\n"
        "# Flashcards\n<details><summary>Q: What is quorum?</summary>A: A majority.</details>\n"
        "# Review\nWhy does a leader need quorum acknowledgement? It prevents stale writes.\n"
        "# ADR Checklist\n- Record the consistency constraint.\n- Explain the failover consequence."
    )
    cleaned = _prepare_for_sectioning(source, BoilerplateCleaner())

    assert "Flashcards" not in cleaned.text
    assert "What is quorum" not in cleaned.text
    assert "Why does a leader need quorum acknowledgement?" in cleaned.text
    assert "ADR Checklist" in cleaned.text
    assert "failover consequence" in cleaned.text


def test_malformed_headings_and_presentation_wrappers_are_removed_without_losing_prose() -> None:
    source = _doc(
        "# }\n"
        "Leader election chooses one coordinator after failure detection.\n"
        "<details><summary>Replication note</summary>Replicas acknowledge committed log entries.</details>\n"
        ".. index:: leader election\n"
        ".. _leader-election-anchor:\n"
        "# [source, bash]\n"
        "The quorum prevents a partitioned minority from committing writes.\n"
        "# {\n"
    )
    cleaned = _prepare_for_sectioning(source, BoilerplateCleaner())

    assert "# }" not in cleaned.text
    assert "# {" not in cleaned.text
    assert "[source, bash]" not in cleaned.text
    assert "<details" not in cleaned.text
    assert "<summary" not in cleaned.text
    assert ".. index::" not in cleaned.text
    assert "Leader election chooses one coordinator" in cleaned.text
    assert "Replicas acknowledge committed log entries" in cleaned.text
    assert "quorum prevents" in cleaned.text


def test_cleanliness_audit_reports_critical_and_warning_classes_deterministically() -> None:
    docs = [
        _doc("TODO: replace this explanation with architecture prose."),
        _doc("Leader election explanation remains useful.", section_title="Flashcards"),
        _doc("# }\nUseful replication prose."),
    ]
    report = cleanliness_audit(docs)

    assert report["critical_count"] == 2
    assert report["warning_count"] == 1
    assert [issue["issue"] for issue in report["issues"]] == [
        "placeholder_marker",
        "training_format_section",
        "malformed_heading_residue",
    ]
