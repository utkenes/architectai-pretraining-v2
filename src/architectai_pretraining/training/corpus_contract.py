"""Validation contract for the canonical Semantic Corpus v3 freeze artifact."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from architectai_pretraining.io import read_jsonl

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
    if manifest.get("artifact_type") != "freeze":
        raise ValueError("DAPT requires artifact_type='freeze'; previews and legacy curated corpora are not valid inputs.")
    if int(manifest.get("semantic_schema_version", 0)) != 3:
        raise ValueError("DAPT requires Semantic Corpus schema version 3.")
    if not str(manifest.get("corpus_version", "")).startswith("architecture-corpus-v3-"):
        raise ValueError("DAPT requires an architecture-corpus-v3 freeze manifest.")
    if not manifest.get("corpus_fingerprint") or not manifest.get("split_fingerprints"):
        raise ValueError("Semantic freeze manifest is missing corpus or split fingerprints.")
    return SemanticFreezeArtifact(directory, manifest)


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def split_integrity_report(artifact: SemanticFreezeArtifact) -> dict[str, Any]:
    """Check IDs, normalized text, and provenance groups across all three splits."""
    splits = {
        name: read_jsonl(artifact.directory / f"{name}.jsonl")
        for name in ("train", "validation", "heldout")
    }
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
