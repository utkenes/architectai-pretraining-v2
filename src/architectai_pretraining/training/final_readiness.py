"""Stage 4.2 final local readiness report and conservative GO/NO-GO gates."""

import json
from pathlib import Path
from typing import Any

from architectai_pretraining.training.package import sha256_file
from architectai_pretraining.training.readiness import generate_readiness_report


def generate_final_readiness_report(
    curated_dir: str | Path = "data/final/curated", training_dir: str | Path = "data/training",
    archive_path: str | Path = "data/training/architectai_dapt_dataset_v2.zip",
) -> dict[str, Any]:
    curated, training, archive = Path(curated_dir), Path(training_dir), Path(archive_path)
    base = generate_readiness_report(curated, training)
    source_audit = json.loads((curated / "source_audit.json").read_text(encoding="utf-8"))
    license_audit = json.loads((curated / "license_audit.json").read_text(encoding="utf-8"))
    train_packed = json.loads((training / "packed" / "train_manifest.json").read_text(encoding="utf-8"))
    validation_packed = json.loads((training / "packed" / "validation_manifest.json").read_text(encoding="utf-8"))
    warnings: list[str] = []
    categories = base["corpus_distribution"]["category"]
    for category in ("adr", "domain_driven_design", "reliability"):
        if category not in categories:
            warnings.append(f"{category} has zero final tokens.")
    if validation_packed["statistics"]["packing_efficiency"] < 0.98:
        warnings.append("Validation packing is below 98% because its final 2048-token sequence is padded; no tokens were dropped.")
    blockers = list(base["blocking_issues"])
    if license_audit["unverified_final_documents"]:
        blockers.append("Unverified license evidence in final corpus.")
    if not archive.is_file():
        blockers.append("Immutable Stage 4.2 dataset archive is missing.")
    payload: dict[str, Any] = {
        **base,
        "source_recovery": source_audit,
        "license_audit": license_audit,
        "packing": {"sequence_length": 2048, "train": train_packed, "validation": validation_packed},
        "dataset_archive": {"version": "architectai_dapt_dataset_v2", "filename": archive.name, "sha256": sha256_file(archive) if archive.is_file() else None},
        "training_harness": {"full_parameter": True, "lora": True, "qlora": True, "checkpoint": True, "resume": True, "metrics_logging": True, "gpu_preflight": True},
        "known_warnings": warnings,
        "blocking_issues": blockers,
        "READY_FOR_COLAB": not blockers,
        "READY_TO_RUN_REAL_BASELINE": not blockers,
        "READY_FOR_STAGE_5A_REAL_SMOKE": not blockers,
        "GO_FOR_FULL_DAPT": False,
    }
    training.mkdir(parents=True, exist_ok=True)
    (training / "final_readiness_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Stage 4.2 Final Training Readiness", ""]
    for key in ("READY_FOR_COLAB", "READY_TO_RUN_REAL_BASELINE", "READY_FOR_STAGE_5A_REAL_SMOKE", "GO_FOR_FULL_DAPT"):
        lines.append(f"{key}={str(payload[key]).lower()}")
    lines.extend(["", "## Blockers", ""] + ([f"- {value}" for value in blockers] or ["- None"]))
    lines.extend(["", "## Quality warnings", ""] + ([f"- {value}" for value in warnings] or ["- None"]))
    (training / "final_readiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
