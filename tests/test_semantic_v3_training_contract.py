"""Integration tests for Semantic v3 freeze-to-DAPT contracts."""

import json

import pytest

from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.training.corpus_contract import (
    load_semantic_freeze,
    split_integrity_report,
    validate_packing_tokenizer,
)
from architectai_pretraining.training.package import create_dataset_package
from architectai_pretraining.training.readiness import generate_readiness_report


def _doc(identifier: str, text: str) -> CorpusDocument:
    return CorpusDocument(
        id=identifier,
        source_id="source",
        source_url="https://example.test/source",
        license_id="MIT",
        category="distributed_systems",
        primary_category="distributed_systems",
        category_hint="distributed_systems",
        extraction_policy="source_policy",
        relative_path=f"{identifier}.md",
        text=text,
        metadata={"provenance_group_id": f"group-{identifier}"},
    )


def _write_freeze(root: object, *, artifact_type: str = "freeze", overlap: bool = False) -> object:
    directory = root / "freeze"  # type: ignore[operator]
    directory.mkdir()
    train = _doc("train", "Raft replication preserves consistency under failures.")
    validation = _doc("validation", "A queue applies backpressure when consumers slow down.")
    heldout = _doc("heldout", "A bounded context protects a domain model boundary.")
    if overlap:
        heldout = train.model_copy(update={"id": "heldout", "metadata": {"provenance_group_id": "group-train"}})
    for name, docs in {"train": [train], "validation": [validation], "heldout": [heldout]}.items():
        (directory / f"{name}.jsonl").write_text("\n".join(doc.to_json_str() for doc in docs) + "\n", encoding="utf-8")
    manifest = {
        "artifact_type": artifact_type,
        "corpus_version": "architecture-corpus-v3-semantic-test",
        "semantic_schema_version": 3,
        "corpus_fingerprint": "corpus-fingerprint",
        "split_fingerprints": {"train": "a", "validation": "b", "heldout": "c"},
        "tokenizer": {"identifier": "Qwen/Qwen3-8B", "revision": "main"},
        "actual_selected_token_count": 100,
        "source_distribution": {"source": {"tokens": 100}},
        "category_distribution": {"distributed_systems": {"tokens": 100}},
        "classification": {"fallback_units": 0, "fallback_tokens": 0, "unresolved_units_rejected": 0},
        "release_eligibility": {"release_eligible_units": 3, "release_ineligible_units": 0},
    }
    (directory / "corpus_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("audit.jsonl", "concept_coverage.json", "category_coverage.json", "source_diagnostics.json", "license_audit.json"):
        (directory / name).write_text("{}\n", encoding="utf-8")
    return directory


def test_preview_and_legacy_artifacts_are_rejected(tmp_path: object) -> None:
    preview = _write_freeze(tmp_path, artifact_type="preview")
    with pytest.raises(ValueError, match="artifact_type='freeze'"):
        load_semantic_freeze(preview)
    with pytest.raises(FileNotFoundError, match="legacy"):
        load_semantic_freeze(tmp_path / "legacy")


def test_split_integrity_and_tokenizer_contract_block_unsafe_inputs(tmp_path: object) -> None:
    freeze = _write_freeze(tmp_path, overlap=True)
    artifact = load_semantic_freeze(freeze)
    assert not split_integrity_report(artifact)["valid"]
    with pytest.raises(ValueError, match="Packing tokenizer"):
        validate_packing_tokenizer(artifact, "other-tokenizer", "main")


def test_readiness_uses_strong_contamination_gate(tmp_path: object) -> None:
    freeze = _write_freeze(tmp_path)
    benchmark = tmp_path / "benchmark.jsonl"  # type: ignore[operator]
    benchmark.write_text(
            json.dumps({"id": "benchmark-1", "scenario": "Raft replication preserves consistency under failures.", "question": "Raft replication preserves consistency under failures.", "rubric": {"criteria": "test"}}) + "\n",
        encoding="utf-8",
    )
    report = generate_readiness_report(freeze, tmp_path / "readiness", benchmark)  # type: ignore[operator]
    assert report["benchmark_contamination"]["contaminated_scenarios"] == 1
    assert "Benchmark contamination rate exceeds" in report["blocking_issues"][0]


def test_v3_package_records_freeze_fingerprint(tmp_path: object) -> None:
    freeze = _write_freeze(tmp_path)
    packed = freeze / "packed"  # type: ignore[operator]
    packed.mkdir()
    for split in ("train", "validation"):
        (packed / f"{split}.jsonl").write_text("{}\n", encoding="utf-8")
        (packed / f"{split}_manifest.json").write_text(
            json.dumps({"fingerprint": f"{split}-packed", "statistics": {"sequence_length": 32}}),
            encoding="utf-8",
        )
    package = create_dataset_package(freeze, tmp_path / "dataset.zip", build_git_sha="test")  # type: ignore[operator]
    assert package["dataset_version"] == "architectai_dapt_dataset_v3"
    assert package["corpus_fingerprint"] == "corpus-fingerprint"
    assert package["packed_fingerprints"]["train"] == "train-packed"
