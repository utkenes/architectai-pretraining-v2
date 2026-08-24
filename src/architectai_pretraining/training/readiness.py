"""Machine-readable readiness gates for a canonical Semantic Corpus v3 freeze."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from architectai_pretraining.benchmark.contamination import check_benchmark_against_corpus
from architectai_pretraining.benchmark.dataset import load_benchmark_dataset
from architectai_pretraining.corpus_v2 import cleanliness_audit
from architectai_pretraining.io import read_jsonl
from architectai_pretraining.semantic import CANONICAL_CATEGORIES
from architectai_pretraining.training.corpus_contract import (
    load_semantic_freeze,
    split_integrity_report,
    validate_packed_artifacts,
)


def generate_readiness_report(
    curated_dir: str | Path,
    output_dir: str | Path = "data/training",
    benchmark_path: str | Path = "data/benchmark/architectai_v1.jsonl",
    max_contamination_rate: float = 0.0,
) -> dict[str, Any]:
    """Evaluate freeze, split, license, and benchmark gates without training."""
    artifact = load_semantic_freeze(curated_dir)
    output = Path(output_dir)
    manifest = artifact.manifest
    splits = {name: read_jsonl(artifact.directory / f"{name}.jsonl") for name in ("train", "validation", "heldout")}
    split_integrity = split_integrity_report(artifact)
    cleanliness = cleanliness_audit([doc for docs in splits.values() for doc in docs])
    invalid = [
        doc.id
        for docs in splits.values()
        for doc in docs
        if not doc.primary_category
        or doc.primary_category not in CANONICAL_CATEGORIES
        or not doc.source_id
        or not doc.relative_path
        or not doc.extraction_policy
        or "benchmark" in doc.source_id.casefold()
    ]
    benchmark_file = Path(benchmark_path)
    benchmark_exists = benchmark_file.is_file()
    contamination = (
        check_benchmark_against_corpus(load_benchmark_dataset(benchmark_file), artifact.directory)
        if benchmark_exists
        else None
    )
    blockers: list[str] = []
    try:
        packed_manifests = validate_packed_artifacts(artifact)
        packed = dict.fromkeys(packed_manifests, True)
    except (FileNotFoundError, ValueError) as error:
        packed_manifests = {}
        packed = dict.fromkeys(("train", "validation"), False)
        blockers.append(f"Packed data integrity gate failed: {error}")
    if not split_integrity["valid"]:
        blockers.append("Train/validation/heldout split isolation failed.")
    if cleanliness["critical_count"]:
        blockers.append(
            f"Corpus cleanliness audit found {cleanliness['critical_count']} critical violations."
        )
    if invalid:
        blockers.append(f"Invalid or unresolved Semantic v3 records: {len(invalid)}.")
    if not benchmark_exists:
        blockers.append("Benchmark dataset is required for the contamination gate.")
    elif contamination and contamination.contamination_rate > max_contamination_rate:
        blockers.append("Benchmark contamination rate exceeds the configured threshold.")
    if manifest.get("classification", {}).get("unresolved_units_rejected") is None:
        blockers.append("Freeze manifest lacks unresolved-classification audit metadata.")
    if not manifest.get("release_eligibility"):
        blockers.append("Freeze manifest lacks release-eligibility metadata.")
    contamination_payload = {
        "total_benchmark_scenarios": contamination.total_scenarios if contamination else 0,
        "contaminated_scenarios": contamination.contaminated_scenarios if contamination else 0,
        "contamination_rate": contamination.contamination_rate if contamination else None,
        "flagged_items_count": len(contamination.flagged_items) if contamination else 0,
        "flagged_items": contamination.flagged_items if contamination else [],
        "detection_policy": {
            "exact_benchmark_text": True,
            "ngram_jaccard_threshold": 0.50,
            "benchmark_containment_threshold": 0.80,
            "minimum_benchmark_ngrams": 5,
            "minimum_matching_ngrams": 4,
        },
        "threshold": max_contamination_rate,
        "passed": bool(contamination and contamination.contamination_rate <= max_contamination_rate),
    }
    categories = manifest.get("category_distribution", {})
    warnings = [f"{category} has zero final tokens." for category in CANONICAL_CATEGORIES if category not in categories]
    payload: dict[str, Any] = {
        "semantic_freeze": {"directory": str(artifact.directory), "artifact_type": manifest["artifact_type"], "corpus_version": manifest["corpus_version"], "semantic_schema_version": manifest["semantic_schema_version"], "corpus_fingerprint": artifact.corpus_fingerprint},
        "tokenizer_consistency": manifest["tokenizer"],
        "corpus_integrity": {"invalid_records": len(invalid), "selected_tokens": manifest["actual_selected_token_count"], "classification": manifest["classification"]},
        "corpus_distribution": {"source": manifest["source_distribution"], "category": categories},
        "licensing": manifest["release_eligibility"],
        "split_integrity": split_integrity,
        "cleanliness_audit": cleanliness,
        "benchmark_contamination": contamination_payload,
        "packing_statistics": {
            "packed_train_valid": packed["train"],
            "packed_validation_valid": packed["validation"],
            "sequence_length": packed_manifests.get("train", {}).get("statistics", {}).get("sequence_length"),
        },
        "known_warnings": warnings,
        "blocking_issues": blockers,
        "READY_FOR_COLAB_BASELINE": not blockers,
        "READY_FOR_STAGE_5A_SMOKE": not blockers and all(packed.values()),
        "GO_FOR_FULL_DAPT": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "readiness_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# ArchitectAI Semantic v3 Training Readiness", ""]
    for key in ("READY_FOR_COLAB_BASELINE", "READY_FOR_STAGE_5A_SMOKE", "GO_FOR_FULL_DAPT"):
        lines.append(f"{key}={str(payload[key]).lower()}")
    lines.extend(["", "## Blocking Issues", ""] + ([f"- {item}" for item in blockers] or ["- None"]))
    (output / "readiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
