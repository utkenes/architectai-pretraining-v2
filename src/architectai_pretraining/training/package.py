"""Reproducible, checksum-verified Colab dataset packages."""

import hashlib
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from architectai_pretraining.training.corpus_contract import (
    SemanticFreezeArtifact,
    load_semantic_freeze,
    sha256_file,
)


def current_git_head() -> str:
    """Return the exact repository commit used to create an immutable package."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("Cannot create a production dataset package without git HEAD.")
    return result.stdout.strip()


def create_dataset_package(
    curated_dir: str | Path,
    output_path: str | Path,
    build_git_sha: str | None = None,
    dataset_version: str = "architectai_dapt_dataset_v3",
    training_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Package an immutable Semantic v3 freeze and its independent packed splits."""
    resolved_git_sha = build_git_sha or current_git_head()
    artifact: SemanticFreezeArtifact = load_semantic_freeze(curated_dir)
    source = artifact.directory
    training = Path(training_dir) if training_dir else source / "packed"
    required = [
        "train.jsonl", "validation.jsonl", "heldout.jsonl", "corpus_manifest.json", "audit.jsonl",
        "concept_coverage.json", "category_coverage.json", "source_diagnostics.json", "license_audit.json",
    ]
    manifest = artifact.manifest
    present = required
    file_hashes = {name: sha256_file(source / name) for name in present}
    training_files = ["train.jsonl", "validation.jsonl", "train_manifest.json", "validation_manifest.json"]
    present_training = [name for name in training_files if (training / name).is_file()]
    if len(present_training) != len(training_files):
        missing_packed = sorted(set(training_files) - set(present_training))
        raise FileNotFoundError(
            "Cannot package a Semantic v3 freeze before deterministic train/validation packing: "
            + ", ".join(missing_packed)
        )
    file_hashes.update({f"packed/{name}": sha256_file(training / name) for name in present_training})
    packed_manifests = {
        name: json.loads((training / f"{name}_manifest.json").read_text(encoding="utf-8"))
        for name in ("train", "validation")
        if (training / f"{name}_manifest.json").is_file()
    }
    package_manifest: dict[str, Any] = {
        "dataset_version": dataset_version,
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_version": manifest["corpus_version"],
        "semantic_schema_version": manifest["semantic_schema_version"],
        "corpus_fingerprint": artifact.corpus_fingerprint,
        "split_fingerprints": manifest["split_fingerprints"],
        "tokenizer": manifest["tokenizer"],
        "build_git_sha": resolved_git_sha,
        "sequence_length": next(iter(packed_manifests.values()), {}).get("statistics", {}).get("sequence_length"),
        "packed_fingerprints": {name: value["fingerprint"] for name, value in packed_manifests.items()},
        "audit_hashes": {
            name: file_hashes[name]
            for name in ("license_audit.json", "source_diagnostics.json", "concept_coverage.json", "category_coverage.json")
        },
        "files": file_hashes,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in present:
            archive.write(source / name, arcname=name)
        for name in present_training:
            archive.write(training / name, arcname=f"packed/{name}")
        archive.writestr("dataset_manifest.json", json.dumps(package_manifest, indent=2))
    package_manifest["package_sha256"] = sha256_file(target)
    Path(f"{target}.sha256").write_text(f"{package_manifest['package_sha256']}  {target.name}\n", encoding="utf-8")
    return package_manifest


def verify_dataset_package(package_path: str | Path) -> dict[str, Any]:
    """Verify an extracted archive's internal manifest and file checksums, else abort."""
    target = Path(package_path)
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read("dataset_manifest.json"))
        failures: list[str] = []
        for name, expected in manifest["files"].items():
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            if actual != expected:
                failures.append(f"{name}: expected {expected}, got {actual}")
    if failures:
        raise RuntimeError("Dataset verification ABORTED: " + "; ".join(failures))
    return {"verified": True, "package_sha256": sha256_file(target), "manifest": manifest}
