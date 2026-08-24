"""Validation contract for the canonical Semantic Corpus v3 freeze artifact."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from architectai_pretraining.io import read_jsonl
from architectai_pretraining.manifest import compute_corpus_fingerprint
from architectai_pretraining.training.data import compute_packed_fingerprint

DEFAULT_FREEZE_DIR = Path("data/corpus_v3/freeze")
REQUIRED_FREEZE_FILES = (
    "train.jsonl",
    "validation.jsonl",
    "heldout.jsonl",
    "corpus_manifest.json",
    "audit.jsonl",
    "concept_coverage.json",
    "category_coverage.json",
    "source_diagnostics.json",
    "license_audit.json",
)
AUDIT_ARTIFACTS = (
    "audit.jsonl",
    "concept_coverage.json",
    "category_coverage.json",
    "source_diagnostics.json",
    "license_audit.json",
)


@dataclass(frozen=True)
class SemanticFreezeArtifact:
    directory: Path
    manifest: dict[str, Any]

    @property
    def corpus_fingerprint(self) -> str:
        return str(self.manifest["corpus_fingerprint"])


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_semantic_freeze(corpus_dir: str | Path) -> SemanticFreezeArtifact:
    """Require an explicit, immutable Semantic v3 freeze; previews are rejected."""
    directory = Path(corpus_dir)
    missing = [name for name in REQUIRED_FREEZE_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Semantic v3 freeze artifact is incomplete or legacy. Required files missing: "
            + ", ".join(missing)
        )
    manifest = json.loads((directory / "corpus_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Semantic freeze corpus manifest must be a JSON object.")
    if manifest.get("artifact_type") != "freeze":
        raise ValueError("DAPT requires artifact_type='freeze'; previews and legacy curated corpora are not valid inputs.")
    if int(manifest.get("semantic_schema_version", 0)) != 3:
        raise ValueError("DAPT requires Semantic Corpus schema version 3.")
    if not str(manifest.get("corpus_version", "")).startswith("architecture-corpus-v3-"):
        raise ValueError("DAPT requires an architecture-corpus-v3 freeze manifest.")
    if not manifest.get("corpus_fingerprint") or not manifest.get("split_fingerprints"):
        raise ValueError("Semantic freeze manifest is missing corpus or split fingerprints.")
    artifact = SemanticFreezeArtifact(directory, manifest)
    validate_freeze_fingerprints(artifact)
    validate_audit_artifact_hashes(artifact)
    return artifact


def _freeze_splits(artifact: SemanticFreezeArtifact) -> dict[str, list[Any]]:
    return {
        name: read_jsonl(artifact.directory / f"{name}.jsonl")
        for name in ("train", "validation", "heldout")
    }


def validate_freeze_fingerprints(artifact: SemanticFreezeArtifact) -> None:
    """Recompute content fingerprints; a copied manifest cannot bless changed JSONL."""
    config_hash = str(artifact.manifest.get("config_hash", ""))
    splits = _freeze_splits(artifact)
    expected_splits = artifact.manifest.get("split_fingerprints", {})
    if not isinstance(expected_splits, dict):
        raise ValueError("Semantic freeze manifest split_fingerprints must be an object.")
    for name, docs in splits.items():
        expected = expected_splits.get(name)
        actual = compute_corpus_fingerprint(docs, config_hash)
        if expected != actual:
            raise ValueError(
                f"Semantic freeze {name} split fingerprint mismatch: expected {expected}, got {actual}."
            )
    all_documents = [document for docs in splits.values() for document in docs]
    actual_corpus = compute_corpus_fingerprint(all_documents, config_hash)
    if artifact.manifest.get("corpus_fingerprint") != actual_corpus:
        raise ValueError(
            "Semantic freeze corpus fingerprint mismatch: "
            f"expected {artifact.manifest.get('corpus_fingerprint')}, got {actual_corpus}."
        )


def validate_audit_artifact_hashes(artifact: SemanticFreezeArtifact) -> None:
    """Require immutable audit evidence alongside the split-content fingerprints."""
    expected_hashes = artifact.manifest.get("artifact_hashes")
    if not isinstance(expected_hashes, dict):
        raise ValueError("Semantic freeze manifest is missing audit artifact hashes.")
    for name in AUDIT_ARTIFACTS:
        expected = expected_hashes.get(name)
        actual = sha256_file(artifact.directory / name)
        if expected != actual:
            raise ValueError(
                f"Semantic freeze audit artifact hash mismatch for {name}: expected {expected}, got {actual}."
            )


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def split_integrity_report(artifact: SemanticFreezeArtifact) -> dict[str, Any]:
    """Check IDs, normalized text, and provenance groups across all three splits."""
    splits = _freeze_splits(artifact)
    values: dict[str, dict[str, set[str]]] = {}
    for name, docs in splits.items():
        values[name] = {
            "ids": {doc.id for doc in docs},
            "texts": {_normalized(doc.text) for doc in docs},
            "groups": {str(doc.metadata.get("provenance_group_id") or doc.id) for doc in docs},
        }
    pairs = (("train", "validation"), ("train", "heldout"), ("validation", "heldout"))
    overlaps = {
        f"{left}_{right}": {
            "id_overlap": len(values[left]["ids"] & values[right]["ids"]),
            "text_overlap": len(values[left]["texts"] & values[right]["texts"]),
            "provenance_group_overlap": len(values[left]["groups"] & values[right]["groups"]),
        }
        for left, right in pairs
    }
    valid = not any(value for pair in overlaps.values() for value in pair.values())
    return {"valid": valid, "splits": {name: len(docs) for name, docs in splits.items()}, "overlaps": overlaps}


def validate_packing_tokenizer(artifact: SemanticFreezeArtifact, identifier: str, revision: str) -> None:
    expected_identifier = artifact.manifest.get("tokenizer", {}).get("identifier")
    expected_revision = artifact.manifest.get("tokenizer", {}).get("revision")
    if (identifier, revision) != (expected_identifier, expected_revision):
        raise ValueError(
            "Packing tokenizer does not match the frozen corpus manifest: "
            f"expected {expected_identifier}@{expected_revision}, got {identifier}@{revision}."
        )


def split_file_fingerprints(artifact: SemanticFreezeArtifact) -> dict[str, str]:
    return {name: sha256_file(artifact.directory / f"{name}.jsonl") for name in ("train", "validation", "heldout")}


def validate_packed_split(
    artifact: SemanticFreezeArtifact, packed_dir: str | Path, split: str
) -> dict[str, Any]:
    """Verify packed bytes, sequence shape, and binding to this exact frozen split."""
    if split not in {"train", "validation"}:
        raise ValueError(f"Only train and validation may be packed; got {split!r}.")
    directory = Path(packed_dir)
    packed_path = directory / f"{split}.jsonl"
    manifest_path = directory / f"{split}_manifest.json"
    if not packed_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Packed {split} data and manifest are both required.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Packed {split} manifest must be a JSON object.")
    sequences: list[dict[str, list[int]]] = []
    with packed_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                sequence = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Packed {split} JSONL is invalid on line {line_number}.") from error
            if not isinstance(sequence, dict):
                raise ValueError(f"Packed {split} JSONL line {line_number} is not a sequence object.")
            sequences.append(sequence)
    actual_fingerprint = compute_packed_fingerprint(sequences)
    if manifest.get("fingerprint") != actual_fingerprint:
        raise ValueError(
            f"Packed {split} fingerprint mismatch: expected {manifest.get('fingerprint')}, got {actual_fingerprint}."
        )
    statistics = manifest.get("statistics", {})
    if not isinstance(statistics, dict):
        raise ValueError(f"Packed {split} manifest statistics must be an object.")
    sequence_length = statistics.get("sequence_length")
    if not isinstance(sequence_length, int) or sequence_length <= 0:
        raise ValueError(f"Packed {split} manifest has no valid sequence_length.")
    for sequence in sequences:
        if set(sequence) != {"input_ids", "labels", "attention_mask"} or any(
            not isinstance(sequence[key], list) or len(sequence[key]) != sequence_length
            for key in ("input_ids", "labels", "attention_mask")
        ):
            raise ValueError(f"Packed {split} sequence shape does not match its manifest.")
    binding = manifest.get("source_freeze", {})
    if not isinstance(binding, dict):
        raise ValueError(f"Packed {split} source_freeze binding must be an object.")
    expected_tokenizer = artifact.manifest.get("tokenizer", {})
    if (
        binding.get("corpus_fingerprint") != artifact.corpus_fingerprint
        or binding.get("split_fingerprint") != artifact.manifest["split_fingerprints"][split]
        or binding.get("tokenizer") != expected_tokenizer
    ):
        raise ValueError(f"Packed {split} artifact is stale or bound to a different Semantic v3 freeze.")
    return manifest


def validate_packed_artifacts(
    artifact: SemanticFreezeArtifact, packed_dir: str | Path | None = None
) -> dict[str, dict[str, Any]]:
    directory = Path(packed_dir) if packed_dir else artifact.directory / "packed"
    return {name: validate_packed_split(artifact, directory, name) for name in ("train", "validation")}
