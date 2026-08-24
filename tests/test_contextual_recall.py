"""Regression tests for conservative, context-aware corpus recall."""

from pathlib import Path

import pytest

import architectai_pretraining.corpus_v2 as corpus_v2
from architectai_pretraining.code_prose import CodeProseAnalyzer
from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.relevance import (
    ArchitectureRelevanceScore,
    ArchitectureRelevanceScorer,
    DomainRelevanceGate,
)
from architectai_pretraining.scoring import DocumentQualityScore, DocumentQualityScorer
from architectai_pretraining.semantic import annotate_document
from architectai_pretraining.tokenizer import MockTokenCounter


def _doc(
    identifier: str,
    text: str,
    *,
    path: str = "architecture.md",
    index: int = 0,
    title: str | None = None,
    section_title: str | None = None,
) -> CorpusDocument:
    return annotate_document(
        CorpusDocument(
            id=identifier,
            source_id="trusted",
            category="distributed_systems",
            category_hint="distributed_systems",
            title=title,
            section_title=section_title,
            section_headings=[section_title] if section_title else [],
            relative_path=path,
            text=text,
            metadata={"provenance_group_id": path, "section_index": index},
        )
    )


def _write_config(path: Path, *, max_section_tokens: int = 100) -> None:
    path.write_text(
        f"""
corpus_version: architecture-corpus-v3-test
max_section_tokens: {max_section_tokens}
tokenizer: {{identifier: Qwen/Qwen3-8B, revision: main}}
quality: {{min_architecture_relevance_score: 0.40, max_link_ratio: 0.22, max_code_ratio: 0.55, borderline_relevance_score: 0.28, borderline_quality_score: 0.33}}
balancing: {{category_token_targets: {{distributed_systems: 1.0}}}}
sources:
  - {{id: trusted, name: Trusted, category: distributed_systems, enabled: false, license_training_status: approved}}
""",
        encoding="utf-8",
    )


def test_metadata_context_scores_later_headingless_chunk_without_changing_training_text() -> None:
    body = (
        "The replicas acknowledge writes after the elected leader records an entry. "
        "This paragraph explains failure recovery and quorum handling."
    )
    plain = CorpusDocument(id="plain", source_id="trusted", category="distributed_systems", text=body)
    contextual = plain.model_copy(
        update={
            "title": "Replication Strategy",
            "section_title": "Leader Election and Recovery",
            "section_headings": ["Leader Election and Recovery"],
        }
    )

    assert contextual.text == body
    assert ArchitectureRelevanceScorer().score(contextual).score > ArchitectureRelevanceScorer().score(plain).score
    assert DocumentQualityScorer().score(contextual).quality_score > DocumentQualityScorer().score(plain).quality_score


def test_ordinary_documentation_links_pass_but_link_dump_and_install_page_fail() -> None:
    prose_lines = [
        "NATS clustering uses routes to exchange membership and leadership information.",
        "Operators should plan recovery paths before a node becomes unavailable.",
        "Clients use reconnect behavior to preserve service continuity during failover.",
        "The topology constrains latency, availability, and operational complexity.",
        "Monitoring exposes slow consumers before backpressure becomes an incident.",
        "A durable stream requires explicit replication and storage choices.",
        "The design documents why queue groups distribute work across consumers.",
        "Use documented security boundaries when connecting separate deployments.",
        "The service model keeps responsibility boundaries visible to operators.",
        "Recovery testing validates that replicas converge after a network partition.",
        "Reference: [cluster guide](https://example.test/cluster).",
        "Reference: [recovery guide](https://example.test/recovery).",
    ]
    technical = _doc("technical", "\n".join(prose_lines))
    assert ArchitectureRelevanceScorer().score(technical).passed

    dump = _doc("dump", "\n".join(f"- [link {i}](https://example.test/{i})" for i in range(20)))
    assert not DomainRelevanceGate().check(dump).is_relevant
    install = _doc("install", "docker run service\napt-get install service", title="Installation")
    assert not DomainRelevanceGate().check(install).is_relevant


def test_code_dominated_reference_is_a_hard_gate_but_explanatory_prose_with_snippet_is_not() -> None:
    analyzer = CodeProseAnalyzer(max_code_token_ratio=0.55)
    code_only = CorpusDocument(
        id="code", source_id="trusted", category="distributed_systems", text="```python\n" + "x = retry()\n" * 60 + "```"
    )
    prose_with_snippet = CorpusDocument(
        id="prose",
        source_id="trusted",
        category="distributed_systems",
        text=("Replication requires a timeout budget and idempotency policy. " * 30) + "\n```python\nretry()\n```",
    )
    assert analyzer.analyze(code_only, MockTokenCounter()).is_code_dominated
    assert not analyzer.analyze(prose_with_snippet, MockTokenCounter()).is_code_dominated


class _FakeRelevance:
    def __init__(self, min_score: float, max_link_ratio: float) -> None:
        self.min_score = min_score

    def score(self, doc: CorpusDocument) -> ArchitectureRelevanceScore:
        score = 0.52 if "neighbor" in doc.text and "borderline" in doc.text else 0.50 if "neighbor" in doc.text else 0.30
        return ArchitectureRelevanceScore(score, score >= self.min_score, [], 0.0, 1.0)


class _FakeQuality:
    min_document_score = 0.45

    def __init__(self, min_document_score: float) -> None:
        self.min_document_score = min_document_score

    def score(self, doc: CorpusDocument) -> DocumentQualityScore:
        score = 0.55 if "neighbor" in doc.text and "borderline" in doc.text else 0.55 if "neighbor" in doc.text else 0.34
        return DocumentQualityScore(score, "medium" if score >= 0.45 else "low")


@pytest.mark.parametrize(
    ("neighbor_path", "neighbor_text", "max_tokens", "should_rescue"),
    [
        ("one.md", "neighbor Raft replication explains quorum recovery.", 100, True),
        ("other.md", "neighbor Raft replication explains quorum recovery.", 100, False),
        ("one.md", "neighbor storage indexing explains compaction.", 100, False),
        ("one.md", "neighbor Raft replication explains quorum recovery.", 5, False),
    ],
)
def test_contextual_rescue_is_same_document_related_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    neighbor_path: str,
    neighbor_text: str,
    max_tokens: int,
    should_rescue: bool,
) -> None:
    config = tmp_path / "corpus.yaml"
    _write_config(config, max_section_tokens=max_tokens)
    pipeline = corpus_v2.CorpusV2Pipeline(config, token_counter=MockTokenCounter())
    monkeypatch.setattr(corpus_v2, "ArchitectureRelevanceScorer", _FakeRelevance)
    monkeypatch.setattr(corpus_v2, "DocumentQualityScorer", _FakeQuality)
    borderline = _doc("borderline", "borderline Raft replication needs quorum recovery.", path="one.md")
    neighbor = _doc("neighbor", neighbor_text, path=neighbor_path, index=1)
    accepted, assessments = pipeline._evaluate_candidates(
        [borderline, neighbor], {"trusted": pipeline.config.source_configs[0]}
    )

    assert (len(accepted) == 1 and accepted[0].metadata.get("recall_decision") == "rescued_borderline") is should_rescue
    if should_rescue:
        assert accepted[0].metadata["grouped_section_ids"] == ["borderline", "neighbor"]
    else:
        assert any(item.decision == "borderline_rejected" for item in assessments)


def test_rescue_never_bypasses_hard_code_or_domain_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "corpus.yaml"
    _write_config(config)
    pipeline = corpus_v2.CorpusV2Pipeline(config, token_counter=MockTokenCounter())
    monkeypatch.setattr(corpus_v2, "ArchitectureRelevanceScorer", _FakeRelevance)
    monkeypatch.setattr(corpus_v2, "DocumentQualityScorer", _FakeQuality)
    neighbor = _doc("neighbor", "neighbor Raft replication explains quorum recovery.", index=1)
    code_heavy = _doc(
        "borderline-code",
        "borderline Raft replication.\n```python\n" + "retry()\n" * 60 + "```",
    )
    link_dump = _doc(
        "borderline-links",
        "\n".join(f"- [replication {index}](https://example.test/{index})" for index in range(20)),
    )
    accepted, assessments = pipeline._evaluate_candidates(
        [code_heavy, link_dump, neighbor], {"trusted": pipeline.config.source_configs[0]}
    )

    assert [doc.id for doc in accepted] == ["neighbor"]
    assert {item.decision for item in assessments if item.document.id != "neighbor"} == {"hard_rejected"}
