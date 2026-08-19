"""Unit tests for domain relevance gate filtering."""

from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.relevance import DomainRelevanceGate


def test_domain_relevance_gate_accepts_architecture_doc() -> None:
    gate = DomainRelevanceGate()
    doc = CorpusDocument(
        id="keps/sig-architecture/001-kep.md",
        title="KEP-001: Kubernetes Architecture Design",
        text="# KEP-001 Architecture Design\nDetailed component boundaries and API trade-offs.",
        source_id="kubernetes_keps",
        category="cloud_architecture",
        license_id="Apache-2.0",
        language="en",
    )
    res = gate.check(doc)
    assert res.is_relevant is True
    assert res.reason is None


def test_domain_relevance_gate_rejects_contributing_md() -> None:
    gate = DomainRelevanceGate()
    doc = CorpusDocument(
        id="docs/CONTRIBUTING.md",
        title="Contributing to Project",
        text="# How to Contribute\nPlease submit pull requests and follow code guidelines.",
        source_id="test_src",
        category="architecture_patterns",
        license_id="MIT",
        language="en",
    )
    res = gate.check(doc)
    assert res.is_relevant is False
    assert res.category == "project_metadata"


def test_domain_relevance_gate_rejects_changelog() -> None:
    gate = DomainRelevanceGate()
    doc = CorpusDocument(
        id="CHANGELOG.md",
        title="Release Notes v1.0.0",
        text="# Release Notes\n- Fixed bug 123\n- Updated dependency",
        source_id="test_src",
        category="architecture_patterns",
        license_id="MIT",
        language="en",
    )
    res = gate.check(doc)
    assert res.is_relevant is False
