"""Focused tests for the audit-first external architecture corpus workflow."""

import json

import pytest

from architectai_pretraining.corpus_v2 import CorpusV2Pipeline
from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.relevance import ArchitectureRelevanceScorer
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


def test_inventory_and_preview_preserve_external_provenance(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "external" / "reliable"  # type: ignore[operator]
    root.mkdir(parents=True)
    (root / "LICENSE").write_text("MIT License\nPermission is hereby granted, free of charge", encoding="utf-8")
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
    (root / "code.md").write_text("# Code\n\n```python\n" + "x = service.retry()\n" * 100 + "```", encoding="utf-8")
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
    records = [json.loads(line) for line in (output / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
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
    repeat = pipeline.build(500, tmp_path / "preview_repeat", frozen=False)  # type: ignore[operator]
    assert repeat["corpus_hash"] == manifest["corpus_hash"]


def test_architecture_relevance_rejects_link_catalog() -> None:
    scorer = ArchitectureRelevanceScorer(min_score=0.20, max_link_ratio=0.20)
    doc = CorpusDocument(
        id="links", source_id="guide", category="system_design", title="Useful links",
        text="\n".join(f"- [tool {i}](https://example.test/{i})" for i in range(30)),
    )
    result = scorer.score(doc)
    assert result.passed is False
    assert result.link_ratio > 0.20


def test_group_split_keeps_section_siblings_together() -> None:
    docs = [
        CorpusDocument(id="a", source_id="s", category="c", text="prose", metadata={"provenance_group_id": "chapter-1"}),
        CorpusDocument(id="b", source_id="s", category="c", text="prose", metadata={"provenance_group_id": "chapter-1"}),
        CorpusDocument(id="c", source_id="s", category="c", text="prose", metadata={"provenance_group_id": "chapter-2"}),
    ]
    result = GroupCorpusSplitter(seed=3).split(docs)
    memberships = {
        doc.id: split
        for split, split_docs in (("train", result.train_documents), ("validation", result.validation_documents), ("heldout", result.heldout_documents))
        for doc in split_docs
    }
    assert memberships["a"] == memberships["b"]
