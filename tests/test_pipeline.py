"""Integration test for end-to-end CorpusPipeline execution."""

import pytest

from architectai_pretraining.io import read_jsonl
from architectai_pretraining.pipeline import CorpusPipeline, PipelineConfig


def test_pipeline_dry_run(tmp_path: pytest.TempPathFactory) -> None:
    manifest_file = tmp_path / "test_sources.yaml"  # type: ignore[operator]
    manifest_content = """
sources:
  - id: local_test_src
    name: "Local Test Document"
    category: "architecture_patterns"
    enabled: true
    type: "local_directory"
    path: "data/raw/manual"
    license_id: "CC-BY-4.0"
"""
    manifest_file.write_text(manifest_content, encoding="utf-8")

    config = PipelineConfig(manifest_path=manifest_file)
    pipeline = CorpusPipeline(config)
    preview = pipeline.dry_run()

    assert preview["total_sources"] == 1
    assert "sources" in preview
    assert preview["enabled_sources"] == 1


def test_pipeline_end_to_end(tmp_path: pytest.TempPathFactory) -> None:
    manifest_file = tmp_path / "test_sources.yaml"  # type: ignore[operator]
    manifest_content = """
sources:
  - id: local_test_src
    name: "Local Test Document"
    category: "architecture_patterns"
    enabled: true
    type: "local_directory"
    path: "data/raw/manual"
    license_id: "CC-BY-4.0"
"""
    manifest_file.write_text(manifest_content, encoding="utf-8")

    output_dir = tmp_path / "final"  # type: ignore[operator]
    cleaned_dir = tmp_path / "cleaned"  # type: ignore[operator]

    config = PipelineConfig(
        manifest_path=manifest_file,
        raw_dir="data/raw",
        cleaned_dir=cleaned_dir,
        final_dir=output_dir,
        train_ratio=0.75,
        seed=42,
    )

    pipeline = CorpusPipeline(config)
    stats, train_path, val_path = pipeline.run()

    assert train_path.exists()
    assert val_path.exists()

    train_docs = read_jsonl(train_path)
    val_docs = read_jsonl(val_path)

    assert len(train_docs) + len(val_docs) == stats.accepted_documents
    assert stats.accepted_documents > 0

    report = stats.to_formatted_report()
    assert "ArchitectAI Corpus Pipeline Report" in report
