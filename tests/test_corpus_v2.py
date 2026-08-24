"""Focused tests for the audit-first external architecture corpus workflow."""

import json
from dataclasses import replace

import pytest

from architectai_pretraining.corpus_v2 import (
    CorpusV2Pipeline,
    _normalize_structural_headings,
    _sectionize,
    load_corpus_v2_config,
)
from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.relevance import ArchitectureRelevanceScorer
from architectai_pretraining.sources import SourceConfig
from architectai_pretraining.splitter import GroupCorpusSplitter
from architectai_pretraining.tokenizer import MockTokenCounter


def _write_config(path: object) -> None:
    path.write_text(  # type: ignore[union-attr]
        """
corpus_version: test-v2
target_tokens: 1000
seed: 7
max_section_tokens: 100
tokenizer: {identifier: Qwen/Qwen3-8B, revision: main}
quality: {min_architecture_relevance_score: 0.20, max_link_ratio: 0.22, max_code_ratio: 0.30}
balancing:
  max_source_token_share: 1.0
  category_token_targets: {reliability_resilience: 1.0}
split: {train_ratio: 0.90, validation_ratio: 0.05, heldout_ratio: 0.05}
sources:
  - id: reliable
    name: Reliable architecture notes
    category: reliability_resilience
    enabled: true
    type: local_directory
    path: ${ARCHITECT_DATA_DIR}/reliable
    license_id: MIT
    verify_license: true
    allowed_content_types: [markdown]
    include_patterns: ["**/*.md"]
    exclude_patterns: ["README*"]
    source_token_cap: 1000
    source_priority: 1.0
""",
        encoding="utf-8",
    )


def test_inventory_and_preview_preserve_external_provenance(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "external" / "reliable"  # type: ignore[operator]
    root.mkdir(parents=True)
    (root / "LICENSE").write_text(
        "MIT License\nPermission is hereby granted, free of charge", encoding="utf-8"
    )
    prose = "\n".join(
        [
            "# Failure handling",
            "## Context",
            "A distributed service must choose retry and timeout policies under latency constraints.",
            "## Decision",
            "Use idempotency keys and bounded retries because replication failures can otherwise duplicate writes.",
            "## Consequences",
            "The trade-off improves reliability but adds storage and operational complexity.",
        ]
        * 6
    )
    (root / "architecture.md").write_text(prose, encoding="utf-8")
    (root / "README.md").write_text("install this package", encoding="utf-8")
    (root / "code.md").write_text(
        "# Code\n\n```python\n" + "x = service.retry()\n" * 100 + "```", encoding="utf-8"
    )
    config = tmp_path / "corpus.yaml"  # type: ignore[operator]
    _write_config(config)
    monkeypatch.setenv("ARCHITECT_DATA_DIR", str(root.parent))

    pipeline = CorpusV2Pipeline(config, token_counter=MockTokenCounter())
    inventory = pipeline.inventory()
    assert inventory["sources"][0]["root"].endswith("external\\reliable")
    assert inventory["sources"][0]["license_verified"] is True

    output = tmp_path / "preview"  # type: ignore[operator]
    manifest = pipeline.build(500, output, frozen=False)
    assert manifest["frozen"] is False
    assert (output / "audit.jsonl").exists()
    records = [
        json.loads(line)
        for line in (output / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(record["decision"] == "rejected" for record in records)
    docs = []
    for split in ("train", "validation", "heldout"):
        docs.extend(
            CorpusDocument.from_json_str(line)
            for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        )
    assert docs
    assert docs[0].relative_path == "architecture.md"
    assert docs[0].verified_license_id == "MIT"
    assert docs[0].content_sha256
    assert docs[0].token_count
    assert (output / "experimental_manifest.json").exists()
    assert (output / "release_eligible_manifest.json").exists()
    repeat = pipeline.build(500, tmp_path / "preview_repeat", frozen=False)  # type: ignore[operator]
    assert repeat["corpus_hash"] == manifest["corpus_hash"]


def test_architecture_relevance_rejects_link_catalog() -> None:
    scorer = ArchitectureRelevanceScorer(min_score=0.20, max_link_ratio=0.20)
    doc = CorpusDocument(
        id="links",
        source_id="guide",
        category="system_design",
        title="Useful links",
        text="\n".join(f"- [tool {i}](https://example.test/{i})" for i in range(30)),
    )
    result = scorer.score(doc)
    assert result.passed is False
    assert result.link_ratio > 0.20


def test_group_split_keeps_section_siblings_together() -> None:
    docs = [
        CorpusDocument(
            id="a",
            source_id="s",
            category="c",
            text="prose",
            metadata={"provenance_group_id": "chapter-1"},
        ),
        CorpusDocument(
            id="b",
            source_id="s",
            category="c",
            text="prose",
            metadata={"provenance_group_id": "chapter-1"},
        ),
        CorpusDocument(
            id="c",
            source_id="s",
            category="c",
            text="prose",
            metadata={"provenance_group_id": "chapter-2"},
        ),
    ]
    result = GroupCorpusSplitter(seed=3).split(docs)
    memberships = {
        doc.id: split
        for split, split_docs in (
            ("train", result.train_documents),
            ("validation", result.validation_documents),
            ("heldout", result.heldout_documents),
        )
        for doc in split_docs
    }
    assert memberships["a"] == memberships["b"]


def test_experimental_config_enables_all_sources() -> None:
    config = load_corpus_v2_config("configs/corpus_v2.yaml")
    assert len(config.source_configs) == 37
    assert all(source.enabled for source in config.source_configs)
    configured = {source.id: source for source in config.source_configs}
    assert configured["nats_docs"].path.endswith(
        "nats.docs.v2-6a3e52c79b369b06890798a5fe2792eddd801852"
    )  # type: ignore[union-attr]
    assert configured["resilience4j_docs"].category == "reliability_resilience"
    assert configured["madr"].category == "adr_decision_reasoning"
    assert configured["opendatahub_adrs"].release_eligible is True
    assert configured["opendatahub_adrs"].commercial_reuse_permitted is None
    assert configured["context_mapping"].license_id == "CC-BY-SA-4.0"
    assert configured["welcome_to_ddd"].include_patterns == ["README.md"]
    assert configured["mit_6824_lecture_notes"].category_hint == "distributed_systems"
    assert configured["architecture_center"].category_hint == "software_architecture"
    assert configured["mozilla_application_services"].include_patterns[0].startswith("docs/adr/")
    assert configured["debezium_outbox_docs"].category_hint == "messaging_event_driven"
    assert configured["debezium_outbox_docs"].source_token_cap == 18000
    assert configured["debezium_outbox_docs"].license_policy["mode"] == "path_scoped"
    assert configured["dotnet_ddd_domain_events_docs"].license_id == "CC-BY-4.0"
    assert configured["dotnet_ddd_domain_events_docs"].include_patterns == [
        "docs/architecture/microservices/microservice-ddd-cqrs-patterns/**/*.md"
    ]
    assert configured["eventuate_tram_sagas_docs"].license_id == "Apache-2.0"
    assert configured["eventuate_tram_sagas_docs"].include_patterns == ["README.adoc"]
    assert configured["eventuate_tram_sagas_docs"].source_token_cap == 24000
    assert configured["gruelbox_transaction_outbox_guide"].license_id == "Apache-2.0"
    assert configured["gruelbox_transaction_outbox_guide"].include_patterns == ["README.md"]
    assert configured["gruelbox_transaction_outbox_guide"].source_token_cap == 10000
    assert configured["tomorrow_one_transactional_outbox_guide"].license_id == "Apache-2.0"
    assert configured["tomorrow_one_transactional_outbox_guide"].source_token_cap == 8000
    assert configured["contextmapper_bounded_context_docs"].license_id == "MIT"
    assert len(configured["contextmapper_bounded_context_docs"].include_patterns) == 15
    assert configured["contextmapper_bounded_context_docs"].source_token_cap == 12000
    assert configured["contextflow_bounded_context_case_study"].include_patterns == [
        "docs/elan-warranty-domain.md",
        "docs/DDD_RELATIONSHIP_PATTERNS.md",
        "docs/DDD_CREW_COMPARISON.md",
    ]
    assert configured["contextflow_bounded_context_case_study"].license_id == "MIT"
    assert configured["contextflow_bounded_context_case_study"].source_token_cap == 10000
    for source_id in (
        "nats_docs",
        "resilience4j_docs",
        "madr",
        "ad_guidance_tool",
        "opendatahub_adrs",
        "context_mapping",
        "welcome_to_ddd",
    ):
        source = configured[source_id]
        assert source.license_training_status == "approved"
        assert source.license_evidence_path
        assert source.license_policy == {"mode": "repository_wide"}
        assert source.metadata["repository_snapshot"] in source.path  # type: ignore[operator]


def test_rst_headings_are_normalized_and_sectionized() -> None:
    doc = CorpusDocument(
        id="rst",
        source_id="network",
        category="networking_systems",
        title="chapter",
        text=(
            "Network Architecture\n====================\n\n"
            + "Architecture trade-offs affect latency and reliability. " * 20
            + "\n\nFailure Handling\n----------------\n\n"
            + "Retries and timeout decisions protect availability. " * 20
        ),
    )
    normalized = _normalize_structural_headings(doc)
    assert "# Network Architecture" in normalized.text
    assert "# Failure Handling" in normalized.text
    sections = _sectionize(normalized, MockTokenCounter(), max_tokens=60)
    assert len(sections) > 1
    assert {section.section_title for section in sections} >= {
        "Network Architecture",
        "Failure Handling",
    }


def test_freeze_preflight_rejects_insufficient_capacity(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "external" / "reliable"  # type: ignore[operator]
    root.mkdir(parents=True)
    (root / "architecture.md").write_text(
        "# Reliability\n\nRetries and timeouts protect availability." * 20, encoding="utf-8"
    )
    config = tmp_path / "corpus.yaml"  # type: ignore[operator]
    _write_config(config)
    monkeypatch.setenv("ARCHITECT_DATA_DIR", str(root.parent))
    pipeline = CorpusV2Pipeline(config, token_counter=MockTokenCounter())
    with pytest.raises(ValueError, match="Freeze preflight failed"):
        pipeline.build(10_000, tmp_path / "freeze", frozen=True)  # type: ignore[operator]
    assert not (tmp_path / "freeze").exists()  # type: ignore[operator]


def test_capacity_uses_same_document_grouping_and_reports_funnel(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "external" / "reliable"  # type: ignore[operator]
    root.mkdir(parents=True)
    (root / "LICENSE").write_text("MIT License\nPermission is hereby granted", encoding="utf-8")
    section = "Raft consensus needs leader election, quorum, and replication. " * 4
    (root / "architecture.md").write_text(
        f"# One\n\n{section}\n\n# Two\n\n{section}\n\n# Three\n\n{section}",
        encoding="utf-8",
    )
    config = tmp_path / "corpus.yaml"  # type: ignore[operator]
    _write_config(config)
    monkeypatch.setenv("ARCHITECT_DATA_DIR", str(root.parent))
    result = CorpusV2Pipeline(config, token_counter=MockTokenCounter()).capacity()
    funnel = result["funnel"]
    assert funnel["sections_generated"] >= 3
    assert (
        funnel["units_after_same_document_grouping"]
        < funnel["quality_passing_units_before_grouping"]
    )
    source = result["source_capacity"]["reliable"]
    assert source["documents_discovered"] == 1
    assert source["documents_accepted"] == 1
    assert source["documents_rejected"] == 0
    assert source["tokens"] == source["eligible_tokens"]
    assert source["training_units"] == funnel["units_after_dedup"]


def test_concept_aware_selection_prefers_less_dominant_source(tmp_path: pytest.TempPathFactory) -> None:
    config = tmp_path / "corpus.yaml"  # type: ignore[operator]
    _write_config(config)
    pipeline = CorpusV2Pipeline(config, token_counter=MockTokenCounter())
    policies = {
        "dominant": SourceConfig(id="dominant", name="Dominant", category="reliability_resilience"),
        "independent": SourceConfig(id="independent", name="Independent", category="reliability_resilience"),
    }
    pipeline.config = replace(
        pipeline.config,
        category_targets={"reliability_resilience": 1.0},
        source_configs=list(policies.values()),
    )

    def doc(identifier: str, source_id: str) -> CorpusDocument:
        return CorpusDocument(
            id=identifier,
            source_id=source_id,
            category="reliability_resilience",
            primary_category="reliability_resilience",
            related_concepts=["backpressure"],
            text="Backpressure protects a service under load.",
            token_count=10,
        )

    docs = [*(doc(f"dominant-{index}", "dominant") for index in range(3)), doc("independent", "independent")]
    selected, _ = pipeline._select(docs, 10, policies, allow_backfill=False)
    assert selected[0].source_id == "independent"
    assert "concept-aware selection" in selected[0].metadata["balance_selection"]
