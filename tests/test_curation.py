"""Integration tests for Stage 3 CurationPipeline execution."""

import pytest

from architectai_pretraining.curation import CurationConfig, CurationPipeline
from architectai_pretraining.io import write_jsonl
from architectai_pretraining.models import CorpusDocument


def test_curation_pipeline_end_to_end(tmp_path: pytest.TempPathFactory) -> None:
    raw_dir = tmp_path / "raw_accepted"  # type: ignore[operator]
    curated_dir = tmp_path / "curated"  # type: ignore[operator]
    manifest_p = tmp_path / "sources.yaml"  # type: ignore[operator]

    manifest_p.write_text(
        """
sources:
  - id: test_src_1
    name: "Test Source 1"
    category: "architecture_patterns"
    enabled: true
    type: "local_directory"
    path: "data/raw/manual"
    license_id: "CC-BY-4.0"
  - id: test_fixtures_local
    name: "Fixture Source"
    category: "architecture_patterns"
    enabled: true
    type: "local_directory"
    path: "data/raw/manual"
    license_id: "CC-BY-4.0"
    metadata:
      is_test_fixture: true
""",
        encoding="utf-8",
    )

    doc1 = CorpusDocument(
        id="arch_doc_1",
        title="Microservices Architecture Tradeoffs",
        text="# Microservices Architecture Tradeoffs\nDetailed tradeoffs between consistency and availability.",
        source_id="test_src_1",
        category="architecture_patterns",
        license_id="CC-BY-4.0",
    )

    doc_fixture = CorpusDocument(
        id="fixture_doc_1",
        title="Test Fixture Document",
        text="Test fixture text that must be isolated.",
        source_id="test_fixtures_local",
        category="architecture_patterns",
        license_id="CC-BY-4.0",
        metadata={"is_test_fixture": True},
    )

    write_jsonl([doc1, doc_fixture], raw_dir / "train.jsonl")

    config = CurationConfig(
        manifest_path=manifest_p,
        raw_accepted_dir=raw_dir,
        curated_dir=curated_dir,
        use_real_tokenizer=False,  # Test mode using MockTokenCounter
        fallback_allowed_in_prod=True,
    )

    pipeline = CurationPipeline(config)
    manifest, report_text = pipeline.run()

    assert manifest.curated_documents_count == 1
    assert manifest.fixture_excluded_count == 1
    assert manifest.output_corpus_fingerprint is not None
    assert (curated_dir / "curation_manifest.json").exists()
    assert (curated_dir / "corpus_audit_report.md").exists()
    assert (curated_dir / "curation_ledger.jsonl").exists()


