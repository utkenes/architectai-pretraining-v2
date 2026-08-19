"""Regression coverage for Stage 4.1 deterministic training preparation."""

from dataclasses import asdict, replace
from pathlib import Path

from architectai_pretraining.balance import CorpusBalancer
from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.scoring import DocumentQualityScore
from architectai_pretraining.tokenizer import MockTokenCounter
from architectai_pretraining.training.data import pack_documents
from architectai_pretraining.training.runner import (
    adamw_settings,
    discover_lora_targets,
    load_smoke_config,
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
