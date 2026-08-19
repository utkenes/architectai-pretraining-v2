"""Reproducible, checksum-verified Colab dataset packages."""

import hashlib
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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
    dataset_version: str = "architectai_dapt_dataset_v2",
    training_dir: str | Path = "data/training",
) -> dict[str, Any]:
    """Package immutable curated and packed metadata; raw clones stay excluded."""
    resolved_git_sha = build_git_sha or current_git_head()
    source = Path(curated_dir)
    training = Path(training_dir)
    required = ["train.jsonl", "validation.jsonl", "curation_manifest.json"]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot package missing curated artifacts: {', '.join(missing)}")
    manifest = json.loads((source / "curation_manifest.json").read_text(encoding="utf-8"))
    files = required + ["corpus_audit_report.md", "source_audit.json", "source_audit.md", "license_audit.json", "license_audit.md"]
    present = [name for name in files if (source / name).is_file()]
    file_hashes = {name: sha256_file(source / name) for name in present}
    training_files = ["readiness_report.json", "readiness_report.md", "final_readiness_report.json", "final_readiness_report.md", "packed/train_manifest.json", "packed/validation_manifest.json"]
    present_training = [name for name in training_files if (training / name).is_file()]
    file_hashes.update({f"training/{name}": sha256_file(training / name) for name in present_training})
    package_manifest: dict[str, Any] = {
        "dataset_version": dataset_version,
        "created_at": datetime.now(UTC).isoformat(),
        "curated_fingerprint": manifest["output_corpus_fingerprint"],
        "train_fingerprint": sha256_file(source / "train.jsonl"),
        "validation_fingerprint": sha256_file(source / "validation.jsonl"),
        "tokenizer_identifier": manifest["tokenizer_identifier"],
        "tokenizer_revision": manifest["tokenizer_revision"],
        "train_tokens": manifest["train_tokens"],
        "validation_tokens": manifest["validation_tokens"],
        "train_documents": manifest["train_documents_count"],
        "validation_documents": manifest["validation_documents_count"],
        "build_git_sha": resolved_git_sha,
        "sequence_length": 2048,
        "files": file_hashes,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in present:
            archive.write(source / name, arcname=name)
        for name in present_training:
            archive.write(training / name, arcname=f"training/{name}")
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
