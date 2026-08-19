"""Stage 5A smoke-training guardrails, deliberately not a full DAPT runner."""

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def validate_causal_lm_batch(batch: dict[str, list[list[int]]], vocab_size: int) -> None:
    """Fail fast before GPU work for malformed causal-LM batches."""
    input_rows = batch.get("input_ids", [])
    label_rows = batch.get("labels", [])
    if not input_rows or not label_rows or len(input_rows) != len(label_rows):
        raise ValueError("Empty or mismatched causal-LM batch.")
    for row, labels in zip(input_rows, label_rows, strict=True):
        if not row or len(row) != len(labels):
            raise ValueError("Empty sequence or input/label length mismatch.")
        if any(token < 0 or token >= vocab_size for token in row):
            raise ValueError("Invalid token ID detected.")
        if all(label == -100 for label in labels):
            raise ValueError("All labels are -100; batch cannot produce a loss.")


def validate_loss_and_gradient(loss: float, gradient_norm: float, safety_grad_norm: float) -> None:
    """Reject non-finite or pathological gradients, not normal clip-threshold exceedance."""
    if not math.isfinite(loss):
        raise FloatingPointError(f"Aborting smoke training: non-finite loss ({loss}).")
    if not math.isfinite(gradient_norm) or gradient_norm > safety_grad_norm:
        raise FloatingPointError(
            f"Aborting smoke training: invalid/exploding gradient norm ({gradient_norm})."
        )


@dataclass
class CheckpointMetadata:
    base_model_identifier: str
    base_revision: str
    tokenizer_identifier: str
    corpus_fingerprint: str
    packed_train_fingerprint: str
    packed_validation_fingerprint: str
    training_config_hash: str
    global_step: int
    optimizer_state_available: bool
    scheduler_state_available: bool
    git_commit_sha: str


def write_checkpoint_metadata(path: str | Path, metadata: CheckpointMetadata) -> None:
    Path(path).write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")


def normalize_config_for_hash(value: Any) -> Any:
    """Make nested training configuration values canonically JSON serializable.

    ``asdict(SmokeRunConfig)`` deliberately preserves Path fields, because their
    values affect reproducibility.  Convert paths to POSIX strings rather than
    dropping them, and recursively normalize the container types used in YAML
    configuration.
    """
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): normalize_config_for_hash(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_config_for_hash(item) for item in value]
    return value


def config_hash(config: dict[str, Any]) -> str:
    normalized = normalize_config_for_hash(config)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"
