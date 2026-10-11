"""A small, resumable Stage 5A causal-LM smoke runner for Colab execution."""

import json
import math
import random
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from architectai_pretraining.training.smoke import (
    CheckpointMetadata,
    config_hash,
    current_git_sha,
    validate_causal_lm_batch,
    validate_loss_and_gradient,
    write_checkpoint_metadata,
)

Strategy = Literal["full_parameter", "lora", "qlora"]
FULL_DAPT_ONE_EPOCH_STEPS = 54


@dataclass(frozen=True)
class SmokeRunConfig:
    model_identifier: str
    model_revision: str
    tokenizer_identifier: str
    tokenizer_revision: str
    sequence_length: int
    train_path: Path
    validation_path: Path
    output_dir: Path
    strategy: Strategy
    max_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    gradient_accumulation_steps: int
    max_grad_norm: float
    safety_grad_norm: float
    save_steps: int
    save_total_limit: int
    seed: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str] | None
    quality_gate_enabled: bool
    benchmark_dataset_path: Path
    diagnostic_dataset_path: Path | None
    benchmark_results_dir: Path
    benchmark_quantization: str
    benchmark_dtype: str
    benchmark_device: str


def load_smoke_config(path: str | Path, output_dir: str | Path | None = None) -> SmokeRunConfig:
    """Parse only the portable Stage 5A settings; no hardware choice is implicit."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model, tokenizer, data = raw["model"], raw["tokenizer"], raw["data"]
    training, strategy = raw["training"], raw["strategy"]
    lora, evaluation = raw.get("lora", {}), raw.get("evaluation", {})
    checkpointing = raw.get("checkpointing", {})
    mode = strategy.get("mode", "full_parameter")
    if mode not in {"full_parameter", "lora", "qlora"}:
        raise ValueError("strategy.mode must be full_parameter, lora, or qlora.")
    save_steps = int(training.get("save_steps", 0))
    save_total_limit = int(checkpointing.get("save_total_limit", 3))
    if save_steps < 0:
        raise ValueError("training.save_steps must be zero or a positive integer.")
    if save_total_limit < 1:
        raise ValueError("checkpointing.save_total_limit must be at least one.")
    diagnostic_dataset = evaluation.get("diagnostic_dataset_path")
    return SmokeRunConfig(
        model_identifier=model["identifier"], model_revision=model["revision"],
        tokenizer_identifier=tokenizer["identifier"], tokenizer_revision=tokenizer.get("revision", "main"), sequence_length=int(data["sequence_length"]),
        train_path=Path(data["train_path"]), validation_path=Path(data["validation_path"]),
        output_dir=Path(output_dir or training["output_dir"]), strategy=mode,
        max_steps=int(training["max_steps"]), learning_rate=float(training["learning_rate"]),
        warmup_ratio=float(training.get("warmup_ratio", 0.0)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        max_grad_norm=float(training["max_grad_norm"]),
        safety_grad_norm=float(training.get("safety_grad_norm", 1000.0)), seed=int(training["seed"]),
        save_steps=save_steps, save_total_limit=save_total_limit,
        lora_rank=int(lora.get("r", 16)), lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)), target_modules=lora.get("target_modules"),
        quality_gate_enabled=bool(evaluation.get("quality_gate_enabled", True)),
        benchmark_dataset_path=Path(evaluation.get("benchmark_dataset_path", "data/benchmark/architectai_v1.jsonl")),
        diagnostic_dataset_path=Path(diagnostic_dataset) if diagnostic_dataset else None,
        benchmark_results_dir=Path(evaluation.get("benchmark_results_dir", "data/benchmark/results/dapt_gate")),
        benchmark_quantization=str(evaluation.get("benchmark_quantization", "4bit")),
        benchmark_dtype=str(evaluation.get("benchmark_dtype", "float16")),
        benchmark_device=str(evaluation.get("benchmark_device", "cuda")),
    )


def load_packed_sequences(path: str | Path) -> list[dict[str, list[int]]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"Packed dataset is empty: {path}")
    return rows


def discover_lora_targets(model: Any, requested: list[str] | None = None) -> list[str]:
    """Validate requested projections against the loaded architecture at runtime."""
    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    candidates = requested or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    matches = [name for name in candidates if name in module_names]
    missing = [name for name in candidates if name not in module_names]
    if requested and missing:
        raise ValueError(f"Configured LoRA target modules are absent: {', '.join(missing)}")
    if not matches:
        raise ValueError("No compatible LoRA projection modules found in the loaded model.")
    return matches


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_batch(row: dict[str, list[int]], device: Any) -> dict[str, Any]:
    import torch

    return {key: torch.tensor([value], dtype=torch.long, device=device) for key, value in row.items()}


def _write_metric(path: Path, values: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": datetime.now(UTC).isoformat(), **values}) + "\n")


def adamw_settings(config: SmokeRunConfig) -> dict[str, float]:
    """Expose optimizer settings so the YAML-to-optimizer contract is testable."""
    if config.weight_decay < 0:
        raise ValueError("training.weight_decay must be non-negative.")
    return {"lr": config.learning_rate, "weight_decay": config.weight_decay}


def qlora_device_map(gpu_count: int) -> str | dict[str, int]:
    """Select an explicit placement strategy that uses every visible GPU."""
    if gpu_count < 1:
        raise RuntimeError("QLoRA requires at least one CUDA device.")
    # `auto` greedily fills GPU 0 and leaves a second 24 GB 3090 idle for an
    # 8B 4-bit model. `balanced` deliberately splits the model across all GPUs.
    return "balanced" if gpu_count > 1 else {"": 0}


def preferred_qlora_compute_dtype(bf16_supported: bool) -> str:
    """Use BF16 tensor-core math when the CUDA device supports it."""
    return "bfloat16" if bf16_supported else "float16"


def _assigned_cuda_devices(model: Any) -> set[int]:
    """Return CUDA device indices assigned by Accelerate's dispatched model map."""
    assigned: set[int] = set()
    for location in getattr(model, "hf_device_map", {}).values():
        if isinstance(location, int):
            assigned.add(location)
        elif isinstance(location, str) and location.startswith("cuda:"):
            assigned.add(int(location.split(":", 1)[1]))
    return assigned


def validate_qlora_device_placement(model: Any, gpu_count: int) -> None:
    """Fail before training if a multi-GPU run would silently use one GPU."""
    if gpu_count < 2:
        return
    expected = set(range(gpu_count))
    assigned = _assigned_cuda_devices(model)
    if not expected.issubset(assigned):
        raise RuntimeError(
            "QLoRA multi-GPU placement is incomplete: "
            f"expected CUDA devices {sorted(expected)}, found {sorted(assigned)}."
        )



def prepare_full_dapt_config(
    config: SmokeRunConfig, output_dir: str | Path = "outputs/dapt-v3-1epoch", epochs: int = 1,
) -> SmokeRunConfig:
    """Promote verified smoke settings to an explicit whole-epoch DAPT run."""
    if epochs < 1:
        raise ValueError("Full DAPT requires at least one epoch.")
    return replace(
        config,
        max_steps=FULL_DAPT_ONE_EPOCH_STEPS * epochs,
        output_dir=Path(output_dir),
        quality_gate_enabled=False,
    )


def _validate_full_dapt_config(config: SmokeRunConfig) -> None:
    if config.max_steps < FULL_DAPT_ONE_EPOCH_STEPS or config.max_steps % FULL_DAPT_ONE_EPOCH_STEPS:
        raise ValueError(
            f"Full DAPT must use a positive whole-epoch multiple of {FULL_DAPT_ONE_EPOCH_STEPS} optimizer steps."
        )
    if config.quality_gate_enabled:
        raise ValueError("Full DAPT must disable automatic quality gating; run it separately after training.")


def _align_optimizer_state_with_parameters(optimizer: Any) -> None:
    """Place optimizer buffers with their trainable parameters after checkpoint load."""
    import torch

    for parameter, state in optimizer.state.items():
        for state_name, value in list(state.items()):
            if not torch.is_tensor(value):
                continue
            # Adam permits a CPU scalar step counter; keeping it there avoids an
            # unnecessary device transfer while preserving PyTorch's valid case.
            if state_name == "step" and value.ndim == 0 and value.device.type == "cpu":
                continue
            target_dtype = parameter.dtype if value.is_floating_point() else value.dtype
            if value.device != parameter.device or value.dtype != target_dtype:
                state[state_name] = value.to(device=parameter.device, dtype=target_dtype)


def _restore_optimizer_state(optimizer: Any, optimizer_state: dict[str, Any]) -> None:
    """Restore Adam state and enforce parameter-local device and dtype placement."""
    optimizer.load_state_dict(optimizer_state)
    _align_optimizer_state_with_parameters(optimizer)


def _restore_checkpoint_training_state(optimizer: Any, scheduler: Any, checkpoint: Path) -> int:
    """Restore optimizer, scheduler, and global step without resetting optimizer state."""
    import torch

    optimizer_state = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    _restore_optimizer_state(optimizer, optimizer_state)
    scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True))
    return int(json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))["global_step"])


def _save_checkpoint(
    model: Any, optimizer: Any, scheduler: Any, config: SmokeRunConfig, step: int,
    corpus_fingerprint: str, packed_train_fingerprint: str, packed_validation_fingerprint: str,
) -> Path:
    import torch

    checkpoint = config.output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint / "model")
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
    metadata = CheckpointMetadata(
        base_model_identifier=config.model_identifier, base_revision=config.model_revision,
        tokenizer_identifier=config.tokenizer_identifier, corpus_fingerprint=corpus_fingerprint,
        packed_train_fingerprint=packed_train_fingerprint, packed_validation_fingerprint=packed_validation_fingerprint,
        training_config_hash=config_hash(asdict(config)), global_step=step,
        optimizer_state_available=True, scheduler_state_available=True, git_commit_sha=current_git_sha(),
    )
    write_checkpoint_metadata(checkpoint / "checkpoint_metadata.json", metadata)
    return checkpoint


def _checkpoint_step(path: Path) -> int:
    """Extract the optimizer step from a checkpoint directory name."""
    return int(path.name.removeprefix("checkpoint-"))


def _prune_checkpoints(output_dir: Path, keep: int) -> list[Path]:
    """Retain only the most recent complete checkpoints in this run directory."""
    checkpoints = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir() and path.name[11:].isdigit()),
        key=_checkpoint_step,
    )
    removed: list[Path] = []
    for checkpoint in checkpoints[:-keep]:
        shutil.rmtree(checkpoint)
        removed.append(checkpoint)
    return removed


def _cuda_memory_snapshot() -> dict[str, dict[str, int]]:
    """Capture per-GPU allocation telemetry for a dispatched QLoRA model."""
    import torch

    return {
        str(index): {
            "allocated": int(torch.cuda.memory_allocated(index)),
            "reserved": int(torch.cuda.memory_reserved(index)),
        }
        for index in range(torch.cuda.device_count())
    }


def _run_training_core(
    config: SmokeRunConfig, corpus_fingerprint: str, packed_train_fingerprint: str,
    packed_validation_fingerprint: str, resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Run the shared GPU training, validation, checkpoint, and resume workflow."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Smoke training requires CUDA; run this command in Colab after preflight.")
    _seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, validation_rows = load_packed_sequences(config.train_path), load_packed_sequences(config.validation_path)
    gpu_count = torch.cuda.device_count()
    load_kwargs: dict[str, Any] = {"revision": config.model_revision, "trust_remote_code": True}
    if config.strategy == "qlora":
        compute_dtype = (
            torch.bfloat16
            if preferred_qlora_compute_dtype(torch.cuda.is_bf16_supported()) == "bfloat16"
            else torch.float16
        )
        load_kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        load_kwargs["device_map"] = qlora_device_map(gpu_count)
    else:
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = {"": 0}
    model: Any = AutoModelForCausalLM.from_pretrained(config.model_identifier, **load_kwargs)
    if config.strategy == "qlora":
        validate_qlora_device_placement(model, gpu_count)
    input_device = model.get_input_embeddings().weight.device
    model.config.use_cache = False
    if config.strategy in {"lora", "qlora"}:
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )

        if config.strategy == "qlora":
            model = prepare_model_for_kbit_training(model)  # type: ignore[no-untyped-call]
        if resume_from:
            model = PeftModel.from_pretrained(model, Path(resume_from) / "model", is_trainable=True)
        else:
            targets = discover_lora_targets(model, config.target_modules)
            model = get_peft_model(model, LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=targets, task_type="CAUSAL_LM"))
    elif resume_from:
        model = AutoModelForCausalLM.from_pretrained(Path(resume_from) / "model", **load_kwargs)
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    warmup_steps = int(config.max_steps * config.warmup_ratio)

    def _learning_rate_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = max(1, config.max_steps - warmup_steps)
        return max(0.0, (config.max_steps - step) / remaining)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _learning_rate_scale)
    start_step = 0
    if resume_from:
        start_step = _restore_checkpoint_training_state(optimizer, scheduler, Path(resume_from))
    metrics_path = config.output_dir / "metrics.jsonl"
    model.train()
    for step in range(start_step + 1, config.max_steps + 1):
        step_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(config.gradient_accumulation_steps):
            row_index = ((step - 1) * config.gradient_accumulation_steps + micro_step) % len(train_rows)
            row = train_rows[row_index]
            validate_causal_lm_batch({key: [value] for key, value in row.items()}, int(model.config.vocab_size))
            batch = _tensor_batch(row, input_device)
            loss = model(**batch).loss
            raw_loss = float(loss.detach().cpu())
            if not math.isfinite(raw_loss):
                raise FloatingPointError(f"Aborting smoke training: non-finite loss at step {step}.")
            step_losses.append(raw_loss)
            (loss / config.gradient_accumulation_steps).backward()
        pre_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm))
        post_norm = min(pre_norm, config.max_grad_norm)
        mean_loss = sum(step_losses) / len(step_losses)
        validate_loss_and_gradient(mean_loss, pre_norm, config.safety_grad_norm)
        optimizer.step()
        scheduler.step()
        _write_metric(metrics_path, {
            "step": step, "loss": mean_loss, "lr": scheduler.get_last_lr()[0],
            "pre_clip_gradient_norm": pre_norm, "post_clip_gradient_norm": post_norm,
            "gpu_memory": _cuda_memory_snapshot(),
        })
        if config.save_steps and step % config.save_steps == 0:
            _save_checkpoint(
                model, optimizer, scheduler, config, step, corpus_fingerprint,
                packed_train_fingerprint, packed_validation_fingerprint,
            )
            _prune_checkpoints(config.output_dir, config.save_total_limit)
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for row in validation_rows[: min(8, len(validation_rows))]:
            losses.append(float(model(**_tensor_batch(row, input_device)).loss.detach().cpu()))
    eval_loss = sum(losses) / len(losses)
    checkpoint = _save_checkpoint(model, optimizer, scheduler, config, config.max_steps, corpus_fingerprint, packed_train_fingerprint, packed_validation_fingerprint)
    _prune_checkpoints(config.output_dir, config.save_total_limit)
    _write_metric(metrics_path, {"step": config.max_steps, "eval_loss": eval_loss})
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint), "global_step": config.max_steps,
        "eval_loss": eval_loss, "strategy": config.strategy,
    }
    if config.quality_gate_enabled:
        if config.strategy == "full_parameter":
            raise RuntimeError(
                "Full-parameter smoke checkpoint was saved, but automatic behavior gating "
                "requires an adapter strategy. Do not promote this run."
            )
        from architectai_pretraining.benchmark.dataset import load_benchmark_dataset
        from architectai_pretraining.benchmark.gate import run_base_vs_finetuned_gate
        from architectai_pretraining.benchmark.models import InferenceConfig

        del model
        torch.cuda.empty_cache()
        gate_config = InferenceConfig(
            model_identifier=config.model_identifier,
            revision=config.model_revision,
            tokenizer_identifier=config.tokenizer_identifier,
            quantization=config.benchmark_quantization,
            dtype=config.benchmark_dtype,
            device=config.benchmark_device,
            enable_thinking=False,
        )
        gate = run_base_vs_finetuned_gate(
            load_benchmark_dataset(config.benchmark_dataset_path), gate_config,
            checkpoint / "model", config.benchmark_results_dir,
        )
        result["quality_gate"] = gate.to_dict()
        if config.diagnostic_dataset_path:
            diagnostic_gate = run_base_vs_finetuned_gate(
                load_benchmark_dataset(config.diagnostic_dataset_path), gate_config,
                checkpoint / "model", config.benchmark_results_dir / "diagnostic",
            )
            result["diagnostic_quality_gate"] = diagnostic_gate.to_dict()
    return result


def run_smoke_training(
    config: SmokeRunConfig, corpus_fingerprint: str, packed_train_fingerprint: str,
    packed_validation_fingerprint: str, resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the guarded 1–50 step Stage 5A smoke workflow."""
    if not 1 <= config.max_steps <= 50:
        raise ValueError("Stage 5A smoke training allows 1–50 steps only.")
    return _run_training_core(
        config, corpus_fingerprint, packed_train_fingerprint, packed_validation_fingerprint, resume_from
    )


def run_full_dapt_training(
    config: SmokeRunConfig, corpus_fingerprint: str, packed_train_fingerprint: str,
    packed_validation_fingerprint: str, resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the explicit one-epoch DAPT promotion with no automatic benchmark gate."""
    _validate_full_dapt_config(config)
    return _run_training_core(
        config, corpus_fingerprint, packed_train_fingerprint, packed_validation_fingerprint, resume_from
    )
