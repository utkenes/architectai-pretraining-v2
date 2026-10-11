"""Regression coverage for Stage 4.1 deterministic training preparation."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from architectai_pretraining.balance import CorpusBalancer
from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.scoring import DocumentQualityScore
from architectai_pretraining.tokenizer import MockTokenCounter
from architectai_pretraining.training.data import pack_documents
from architectai_pretraining.training.runner import (
    _align_optimizer_state_with_parameters,
    _prune_checkpoints,
    _restore_checkpoint_training_state,
    _restore_optimizer_state,
    _validate_full_dapt_config,
    adamw_settings,
    discover_lora_targets,
    load_smoke_config,
    preferred_qlora_compute_dtype,
    prepare_full_dapt_config,
    qlora_device_map,
    run_smoke_training,
    validate_qlora_device_placement,
)
from architectai_pretraining.training.smoke import (
    config_hash,
    normalize_config_for_hash,
    validate_causal_lm_batch,
)


def _doc(identifier: str, source: str, category: str, owner: str) -> CorpusDocument:
    return CorpusDocument(
        id=identifier,
        source_id=source,
        category=category,
        text=f"Architecture document {identifier} has durable technical prose.",
        license_id="MIT",
        metadata={"repository_owner": owner},
    )


def test_balancer_enforces_final_caps_when_feasible() -> None:
    docs = [_doc(f"a{i}", "a", "ca", "oa") for i in range(6)]
    docs += [_doc(f"b{i}", "b", "cb", "ob") for i in range(6)]
    docs += [_doc(f"c{i}", "c", "cc", "oc") for i in range(6)]
    docs += [_doc(f"d{i}", "d", "cd", "od") for i in range(2)]
    tokens = {doc.id: 100 for doc in docs}
    scores = {doc.id: DocumentQualityScore(0.6, "medium") for doc in docs}
    result = CorpusBalancer(0.35, 0.35, 0.35).balance(docs, tokens, scores)
    assert result.constraints_satisfied
    assert result.concentration_after.top_1_source_share <= 0.351
    assert result.concentration_after.top_category_share <= 0.351
    assert result.concentration_after.top_organization_share <= 0.351


def test_balancer_reports_impossible_cap() -> None:
    docs = [_doc("only", "source", "category", "owner")]
    result = CorpusBalancer(0.35, 0.4, 0.4).balance(
        docs, {"only": 100}, {"only": DocumentQualityScore(0.8, "high")}
    )
    assert not result.constraints_satisfied
    assert any(not status.satisfied for status in result.constraint_statuses)


def test_packing_has_eos_boundaries_is_deterministic_and_masks_padding() -> None:
    docs = [_doc("b", "s", "c", "o"), _doc("a", "s2", "c2", "o2")]
    tokenizer = MockTokenCounter()
    first = pack_documents(docs, tokenizer, sequence_length=32)
    second = pack_documents(list(reversed(docs)), tokenizer, sequence_length=32)
    assert first.fingerprint == second.fingerprint
    assert tokenizer.eos_token_id in first.sequences[0]["input_ids"]
    assert all(
        label == -100 for token, label in zip(first.sequences[-1]["attention_mask"], first.sequences[-1]["labels"], strict=True) if token == 0
    )


def test_batch_validator_rejects_all_masked_labels() -> None:
    try:
        validate_causal_lm_batch({"input_ids": [[1, 2]], "labels": [[-100, -100]]}, vocab_size=10)
    except ValueError as error:
        assert "All labels" in str(error)
    else:
        raise AssertionError("Expected invalid masked batch to be rejected")


class _NamedModuleModel:
    def named_modules(self) -> list[tuple[str, object]]:
        return [("layers.0.self_attn.q_proj", object()), ("layers.0.self_attn.v_proj", object())]


def test_lora_target_discovery_validates_runtime_modules() -> None:
    assert discover_lora_targets(_NamedModuleModel(), ["q_proj", "v_proj"]) == ["q_proj", "v_proj"]
    try:
        discover_lora_targets(_NamedModuleModel(), ["not_a_projection"])
    except ValueError as error:
        assert "absent" in str(error)
    else:
        raise AssertionError("Missing target module should fail before training.")


def test_default_dapt_configuration_is_conservative_and_gated() -> None:
    config = load_smoke_config("configs/dapt.yaml")
    assert config.strategy == "qlora"
    assert config.learning_rate <= 0.000005
    assert config.target_modules == ["q_proj", "v_proj"]
    assert config.quality_gate_enabled
    assert config.benchmark_dataset_path.name == "architectai_v1.jsonl"
    assert config.diagnostic_dataset_path is None
    assert config.save_steps == 10
    assert config.save_total_limit == 3


def test_qlora_uses_balanced_multi_gpu_placement_and_bf16_when_available() -> None:
    assert qlora_device_map(1) == {"": 0}
    assert qlora_device_map(2) == "balanced"
    assert preferred_qlora_compute_dtype(True) == "bfloat16"
    assert preferred_qlora_compute_dtype(False) == "float16"


class _DispatchedModel:
    def __init__(self, device_map: dict[str, int]) -> None:
        self.hf_device_map = device_map


def test_qlora_multi_gpu_placement_requires_each_visible_gpu() -> None:
    validate_qlora_device_placement(_DispatchedModel({"model.layers.0": 0, "model.layers.1": 1}), 2)
    with pytest.raises(RuntimeError, match="placement is incomplete"):
        validate_qlora_device_placement(_DispatchedModel({"model.layers.0": 0}), 2)


def test_checkpoint_retention_keeps_only_the_latest_complete_checkpoints(tmp_path: Path) -> None:
    for step in (10, 20, 30, 40):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "checkpoint_metadata.json").write_text("{}", encoding="utf-8")
    removed = _prune_checkpoints(tmp_path, keep=3)
    assert [path.name for path in removed] == ["checkpoint-10"]
    assert sorted(path.name for path in tmp_path.glob("checkpoint-*") if path.is_dir()) == [
        "checkpoint-20", "checkpoint-30", "checkpoint-40",
    ]


def test_adamw_settings_use_weight_decay_from_yaml() -> None:
    config = load_smoke_config("configs/dapt.yaml")
    assert adamw_settings(config) == {"lr": config.learning_rate, "weight_decay": 0.01}


def test_checkpoint_config_hash_serializes_actual_smoke_config_paths_deterministically() -> None:
    config = load_smoke_config("configs/dapt.yaml")
    config_dict = asdict(config)
    first = config_hash(config_dict)
    second = config_hash(config_dict)
    changed_path = replace(config, train_path=Path("data/training/packed/other-train.jsonl"))
    assert first == second
    assert first != config_hash(asdict(changed_path))
    assert normalize_config_for_hash(
        {"paths": [Path("one/two"), (Path("three/four"),)]}
    ) == {"paths": ["one/two", ["three/four"]]}


def test_optimizer_checkpoint_restore_aligns_state_with_resumed_parameter_device() -> None:
    import torch

    source_parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    source_optimizer = torch.optim.AdamW([source_parameter])
    source_parameter.grad = torch.ones_like(source_parameter)
    source_optimizer.step()

    resumed_parameter = torch.nn.Parameter(torch.empty(4, device="meta", dtype=torch.float16))
    resumed_optimizer = torch.optim.AdamW([resumed_parameter])
    _restore_optimizer_state(resumed_optimizer, source_optimizer.state_dict())

    restored = resumed_optimizer.state[resumed_parameter]
    assert restored["exp_avg"].device == resumed_parameter.device
    assert restored["exp_avg_sq"].device == resumed_parameter.device
    assert restored["exp_avg"].dtype == resumed_parameter.dtype
    assert restored["exp_avg_sq"].dtype == resumed_parameter.dtype
    assert restored["step"].device.type == "cpu"


def test_checkpoint_restore_preserves_optimizer_scheduler_and_global_step(tmp_path: Path) -> None:
    import torch

    source_parameter = torch.nn.Parameter(torch.ones(2))
    source_optimizer = torch.optim.AdamW([source_parameter], lr=0.1)
    source_scheduler = torch.optim.lr_scheduler.LambdaLR(source_optimizer, lambda _: 1.0)
    source_parameter.grad = torch.ones_like(source_parameter)
    source_optimizer.step()
    source_scheduler.step()
    torch.save(source_optimizer.state_dict(), tmp_path / "optimizer.pt")
    torch.save(source_scheduler.state_dict(), tmp_path / "scheduler.pt")
    (tmp_path / "trainer_state.json").write_text(json.dumps({"global_step": 10}), encoding="utf-8")

    resumed_parameter = torch.nn.Parameter(torch.ones(2))
    resumed_optimizer = torch.optim.AdamW([resumed_parameter], lr=0.1)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _: 1.0)
    restored_step = _restore_checkpoint_training_state(resumed_optimizer, resumed_scheduler, tmp_path)

    assert restored_step == 10
    assert resumed_optimizer.state[resumed_parameter]["exp_avg"].numel() == 2
    assert resumed_optimizer.state[resumed_parameter]["step"].item() == 1
    assert resumed_scheduler.state_dict()["last_epoch"] == source_scheduler.state_dict()["last_epoch"]


def test_optimizer_alignment_keeps_cpu_step_scalar_unchanged() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    step = torch.tensor(10.0, dtype=torch.float64)
    optimizer.state[parameter] = {"step": step, "exp_avg": torch.ones(2), "exp_avg_sq": torch.ones(2)}
    _align_optimizer_state_with_parameters(optimizer)

    assert optimizer.state[parameter]["step"].device.type == "cpu"
    assert optimizer.state[parameter]["step"].dtype == torch.float64

def test_smoke_rejects_more_than_fifty_steps_before_loading_a_model() -> None:
    config = replace(load_smoke_config("configs/dapt.yaml"), max_steps=51)

    with pytest.raises(ValueError, match="1–50"):
        run_smoke_training(config, "corpus", "train", "validation")


def test_full_dapt_promotion_is_fixed_at_one_epoch_and_disables_automatic_gate() -> None:
    smoke_config = load_smoke_config("configs/dapt.yaml")
    full_config = prepare_full_dapt_config(smoke_config)

    _validate_full_dapt_config(full_config)
    assert full_config.max_steps == 54
    assert full_config.output_dir == Path("outputs/dapt-v3-1epoch")
    assert not full_config.quality_gate_enabled
    assert full_config.learning_rate == smoke_config.learning_rate == 0.000005
    assert full_config.gradient_accumulation_steps == smoke_config.gradient_accumulation_steps == 8
    assert full_config.lora_rank == smoke_config.lora_rank == 8
    assert full_config.lora_alpha == smoke_config.lora_alpha == 16
    assert full_config.target_modules == smoke_config.target_modules == ["q_proj", "v_proj"]


def test_full_dapt_promotion_accepts_explicit_whole_epoch_extension() -> None:
    full_config = prepare_full_dapt_config(load_smoke_config("configs/dapt.yaml"), epochs=2)

    _validate_full_dapt_config(full_config)
    assert full_config.max_steps == 108


def test_full_dapt_rejects_any_step_count_other_than_one_epoch() -> None:
    config = replace(prepare_full_dapt_config(load_smoke_config("configs/dapt.yaml")), max_steps=53)

    with pytest.raises(ValueError, match="whole-epoch multiple"):
        _validate_full_dapt_config(config)
