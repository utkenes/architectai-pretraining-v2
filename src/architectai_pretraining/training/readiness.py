"""Machine-readable Stage 4.1 readiness report generation."""

import json
from pathlib import Path
from typing import Any

from architectai_pretraining.io import read_jsonl
from architectai_pretraining.manifest import CurationManifest, compute_corpus_fingerprint


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def generate_readiness_report(
    curated_dir: str | Path,
    output_dir: str | Path = "data/training",
    benchmark_path: str | Path = "data/benchmark/architectai_v1.jsonl",
) -> dict[str, Any]:
    """Audit required correctness gates without declaring full DAPT readiness."""
    curated = Path(curated_dir)
    output = Path(output_dir)
    manifest = CurationManifest(**json.loads((curated / "curation_manifest.json").read_text(encoding="utf-8")))
    train, validation = read_jsonl(curated / "train.jsonl"), read_jsonl(curated / "validation.jsonl")
    train_ids, validation_ids = {doc.id for doc in train}, {doc.id for doc in validation}
    train_texts, validation_texts = {_normalized(doc.text) for doc in train}, {_normalized(doc.text) for doc in validation}
    exact_id_overlap = len(train_ids & validation_ids)
    exact_text_overlap = len(train_texts & validation_texts)
    invalid = [
        doc.id
        for doc in train + validation
        if not doc.id
        or not doc.text.strip()
        or not doc.source_id
        or not doc.category
        or not doc.language
        or not doc.source_url
        or not (doc.license_id or doc.metadata.get("verified_license_id"))
        or "benchmark" in doc.source_id.lower()
        or "benchmark" in str(doc.source_url).lower()
    ]
    benchmark_exists = Path(benchmark_path).is_file()
    benchmark_content = Path(benchmark_path).read_text(encoding="utf-8") if benchmark_exists else ""
    benchmark_contamination = sum(1 for text in train_texts | validation_texts if text and text in _normalized(benchmark_content))
    packed_train = curated.parent.parent / "training" / "packed" / "train_manifest.json"
    packed_validation = curated.parent.parent / "training" / "packed" / "validation_manifest.json"
    blockers: list[str] = []
    if exact_id_overlap or exact_text_overlap:
        blockers.append("Train/validation exact overlap detected.")
    if invalid:
        blockers.append(f"Invalid curated records: {len(invalid)}.")
    if benchmark_contamination:
        blockers.append("Benchmark contamination candidates detected.")
    if not manifest.output_corpus_fingerprint:
        blockers.append("Curated fingerprint missing.")
    payload: dict[str, Any] = {
        "repository_integrity": {"curated_fingerprint": manifest.output_corpus_fingerprint},
        "tokenizer_consistency": {
            "identifier": manifest.tokenizer_identifier,
            "revision": manifest.tokenizer_revision,
            "qwen3_only_production_default": manifest.tokenizer_identifier == "Qwen/Qwen3-8B",
        },
        "corpus_integrity": {"invalid_records": len(invalid), "curated_documents": manifest.curated_documents_count},
        "corpus_distribution": {"source": manifest.source_distribution, "category": manifest.category_distribution},
        "licensing": {"distribution": manifest.license_distribution},
        "train_validation_leakage": {
            "train_val_exact_id_overlap": exact_id_overlap,
            "train_val_exact_text_overlap": exact_text_overlap,
            "train_fingerprint": compute_corpus_fingerprint(train, "train"),
            "validation_fingerprint": compute_corpus_fingerprint(validation, "validation"),
        },
        "benchmark_isolation": {"benchmark_exists": benchmark_exists, "benchmark_contamination_candidates": benchmark_contamination},
        "packing_statistics": {"packed_train_manifest_exists": packed_train.is_file(), "packed_validation_manifest_exists": packed_validation.is_file()},
        "model_compatibility": {"model": "Qwen/Qwen3-8B", "full_parameter_8b_on_t4_guaranteed": False},
        "training_dependency_readiness": {"config_exists": Path("configs/dapt.yaml").is_file()},
        "colab_readiness": {"dataset_package_required": True},
        "known_warnings": [],
        "blocking_issues": blockers,
        "READY_FOR_COLAB_BASELINE": benchmark_exists,
        "READY_FOR_STAGE_5A_SMOKE": not blockers and Path("configs/dapt.yaml").is_file(),
        "GO_FOR_FULL_DAPT": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "readiness_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# ArchitectAI Stage 4.1 Training Readiness", ""]
    for key in ("READY_FOR_COLAB_BASELINE", "READY_FOR_STAGE_5A_SMOKE", "GO_FOR_FULL_DAPT"):
        lines.append(f"{key}={str(payload[key]).lower()}")
    lines.extend(["", "## Blocking Issues", ""])
    lines.extend([f"- {item}" for item in blockers] or ["- None"])
    (output / "readiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
