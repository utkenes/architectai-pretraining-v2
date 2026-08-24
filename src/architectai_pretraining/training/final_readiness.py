"""Final conservative readiness report for a Semantic Corpus v3 dataset package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from architectai_pretraining.training.corpus_contract import load_semantic_freeze
from architectai_pretraining.training.package import sha256_file
from architectai_pretraining.training.readiness import generate_readiness_report


def generate_final_readiness_report(
    curated_dir: str | Path = "data/corpus_v3/freeze",
    training_dir: str | Path = "data/training",
    archive_path: str | Path = "data/training/architectai_dapt_dataset_v3.zip",
) -> dict[str, Any]:
    artifact = load_semantic_freeze(curated_dir)
    training, archive = Path(training_dir), Path(archive_path)
    base = generate_readiness_report(artifact.directory, training)
    packed_dir = artifact.directory / "packed"
    blockers = list(base["blocking_issues"])
    packed: dict[str, Any] = {}
    for split in ("train", "validation"):
        manifest = packed_dir / f"{split}_manifest.json"
        if not manifest.is_file():
            blockers.append(f"Packed {split} manifest is missing.")
        else:
            packed[split] = json.loads(manifest.read_text(encoding="utf-8"))
    if not archive.is_file():
        blockers.append("Immutable Semantic v3 dataset archive is missing.")
    payload: dict[str, Any] = {
        **base,
        "packing": {"sequence_length": packed.get("train", {}).get("statistics", {}).get("sequence_length"), **packed},
        "dataset_archive": {"version": "architectai_dapt_dataset_v3", "filename": archive.name, "sha256": sha256_file(archive) if archive.is_file() else None},
        "training_harness": {"full_parameter": True, "lora": True, "qlora": True, "checkpoint": True, "resume": True, "metrics_logging": True, "gpu_preflight": True},
        "blocking_issues": blockers,
        "READY_FOR_COLAB": not blockers,
        "READY_TO_RUN_REAL_BASELINE": not blockers,
        "READY_FOR_STAGE_5A_REAL_SMOKE": not blockers,
        "GO_FOR_FULL_DAPT": False,
    }
    training.mkdir(parents=True, exist_ok=True)
    (training / "final_readiness_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
