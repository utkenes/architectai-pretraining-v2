"""Integration tests for Semantic v3 freeze-to-DAPT contracts."""

import json
from pathlib import Path

import pytest

from architectai_pretraining.manifest import compute_corpus_fingerprint
from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.tokenizer import MockTokenCounter
from architectai_pretraining.training.corpus_contract import (
    load_semantic_freeze,
    sha256_file,
    split_integrity_report,
    validate_packed_artifacts,
    validate_packing_tokenizer,
)
from architectai_pretraining.training.data import pack_documents
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


def _write_freeze(
    root: Path,
    *,
    artifact_type: str = "freeze",
    overlap: bool = False,
    train_text: str = "Raft replication preserves consistency under failures.",
) -> Path:
    directory = root / "freeze"
    directory.mkdir(parents=True)
    train = _doc("train", train_text)
    validation = _doc("validation", "A queue applies backpressure when consumers slow down.")
    heldout = _doc("heldout", "A bounded context protects a domain model boundary.")
    if overlap:
        heldout = train.model_copy(update={"id": "heldout", "metadata": {"provenance_group_id": "group-train"}})
    splits = {"train": [train], "validation": [validation], "heldout": [heldout]}
    for name, docs in splits.items():
        (directory / f"{name}.jsonl").write_text("\n".join(doc.to_json_str() for doc in docs) + "\n", encoding="utf-8")
    for name in (
        "audit.jsonl",
        "concept_coverage.json",
        "category_coverage.json",
        "source_diagnostics.json",
        "license_audit.json",
    ):
        (directory / name).write_text("{}\n", encoding="utf-8")
    config_hash = "semantic-v3-test-config"
    manifest = {
        "artifact_type": artifact_type,
        "corpus_version": "architecture-corpus-v3-semantic-test",
        "semantic_schema_version": 3,
        "config_hash": config_hash,
        "corpus_fingerprint": compute_corpus_fingerprint(
            [doc for docs in splits.values() for doc in docs], config_hash
        ),
        "split_fingerprints": {
            name: compute_corpus_fingerprint(docs, config_hash) for name, docs in splits.items()
        },
        "tokenizer": {"identifier": "Qwen/Qwen3-8B", "revision": "main"},
        "actual_selected_token_count": 100,
        "source_distribution": {"source": {"tokens": 100}},
        "category_distribution": {"distributed_systems": {"tokens": 100}},
        "classification": {"fallback_units": 0, "fallback_tokens": 0, "unresolved_units_rejected": 0},
        "release_eligibility": {"release_eligible_units": 3, "release_ineligible_units": 0},
        "artifact_hashes": {
            name: sha256_file(directory / name)
            for name in (
                "audit.jsonl",
                "concept_coverage.json",
                "category_coverage.json",
                "source_diagnostics.json",
                "license_audit.json",
            )
        },
    }
    (directory / "corpus_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def _write_packed(freeze: Path) -> None:
    artifact = load_semantic_freeze(freeze)
    tokenizer = MockTokenCounter()
    for split in ("train", "validation"):
        documents = [
            CorpusDocument.model_validate_json(line)
            for line in (freeze / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        packed = pack_documents(documents, tokenizer, sequence_length=32)
        packed.write_jsonl(freeze / "packed" / f"{split}.jsonl")
        packed.write_manifest(
            freeze / "packed" / f"{split}_manifest.json",
            source_corpus_fingerprint=artifact.corpus_fingerprint,
            source_split_fingerprint=artifact.manifest["split_fingerprints"][split],
            tokenizer_identifier="Qwen/Qwen3-8B",
            tokenizer_revision="main",
        )


def test_preview_and_legacy_artifacts_are_rejected(tmp_path: Path) -> None:
    preview = _write_freeze(tmp_path, artifact_type="preview")
    with pytest.raises(ValueError, match="artifact_type='freeze'"):
        load_semantic_freeze(preview)
    with pytest.raises(FileNotFoundError, match="legacy"):
        load_semantic_freeze(tmp_path / "legacy")


def test_split_integrity_and_tokenizer_contract_block_unsafe_inputs(tmp_path: Path) -> None:
    freeze = _write_freeze(tmp_path, overlap=True)
    artifact = load_semantic_freeze(freeze)
    assert not split_integrity_report(artifact)["valid"]
    with pytest.raises(ValueError, match="Packing tokenizer"):
        validate_packing_tokenizer(artifact, "other-tokenizer", "main")


@pytest.mark.parametrize("split", ["train", "validation", "heldout"])
def test_freeze_content_fingerprint_rejects_tampered_split(tmp_path: Path, split: str) -> None:
    freeze = _write_freeze(tmp_path)
    target = freeze / f"{split}.jsonl"
    target.write_text(target.read_text(encoding="utf-8").replace(".", " altered."), encoding="utf-8")
    with pytest.raises(ValueError, match=f"{split} split fingerprint mismatch"):
        load_semantic_freeze(freeze)


def test_freeze_audit_hash_rejects_tampering(tmp_path: Path) -> None:
    freeze = _write_freeze(tmp_path)
    (freeze / "license_audit.json").write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="audit artifact hash mismatch"):
        load_semantic_freeze(freeze)


def test_packed_artifacts_bind_content_and_freeze(tmp_path: Path) -> None:
    freeze = _write_freeze(tmp_path)
    _write_packed(freeze)
    artifact = load_semantic_freeze(freeze)
    assert set(validate_packed_artifacts(artifact)) == {"train", "validation"}
    packed_train = freeze / "packed" / "train.jsonl"
    packed_train.write_text(packed_train.read_text(encoding="utf-8").replace("1", "2", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="Packed train fingerprint mismatch"):
        validate_packed_artifacts(artifact)


def test_packed_artifacts_reject_stale_freeze_binding(tmp_path: Path) -> None:
    freeze_a = _write_freeze(tmp_path / "a")
    _write_packed(freeze_a)
    freeze_b = _write_freeze(tmp_path / "b", train_text="A changed frozen train document.")
    with pytest.raises(ValueError, match="stale or bound to a different"):
        validate_packed_artifacts(load_semantic_freeze(freeze_b), freeze_a / "packed")


def test_readiness_uses_strong_contamination_gate(tmp_path: Path) -> None:
    freeze = _write_freeze(tmp_path)
    _write_packed(freeze)
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "id": "benchmark-1",
                "scenario": "Raft replication preserves consistency under failures.",
                "question": "Raft replication preserves consistency under failures.",
                "rubric": {"criteria": "test"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = generate_readiness_report(freeze, tmp_path / "readiness", benchmark)
    assert report["benchmark_contamination"]["contaminated_scenarios"] == 1
    assert any("Benchmark contamination rate exceeds" in issue for issue in report["blocking_issues"])


def test_v3_package_records_freeze_fingerprint(tmp_path: Path) -> None:
    freeze = _write_freeze(tmp_path)
    _write_packed(freeze)
    package = create_dataset_package(freeze, tmp_path / "dataset.zip", build_git_sha="test")
    assert package["dataset_version"] == "architectai_dapt_dataset_v3"
    assert package["corpus_fingerprint"] == load_semantic_freeze(freeze).corpus_fingerprint
    assert package["packed_fingerprints"]["train"]
