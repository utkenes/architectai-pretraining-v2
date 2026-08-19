"""Regression tests for behavioral DAPT promotion gates."""

import json
from pathlib import Path

import pytest

from architectai_pretraining.benchmark.dataset import BenchmarkDataset
from architectai_pretraining.benchmark.gate import (
    compare_base_to_finetuned,
    run_base_vs_finetuned_gate,
    summarize_behavior,
)
from architectai_pretraining.benchmark.models import (
    BenchmarkSample,
    EvaluationResult,
    InferenceConfig,
    RawOutput,
)
from architectai_pretraining.benchmark.runner import BenchmarkRunner


def _output(sample_id: str, text: str) -> RawOutput:
    return RawOutput(sample_id, "Qwen/Qwen3-8B", "hash", text, 1, 1, 0.0)


def _evaluation(sample_id: str, score: float = 0.8) -> EvaluationResult:
    return EvaluationResult(sample_id, "test", "easy", {}, score, 0.0)


def test_gate_rejects_repetition_and_prompt_echo_regression() -> None:
    prompts = {"one": "Explain the tradeoffs for a payment service with a small team."}
    base = summarize_behavior(
        prompts,
        [_output("one", "Use a modular monolith first and revisit microservices as ownership grows.")],
        [_evaluation("one")],
    )
    bad_response = (
        "Explain the tradeoffs for a payment service with a small team "
        "Explain the tradeoffs for a payment service with a small team "
        "Explain the tradeoffs for a payment service with a small team"
    )
    finetuned = summarize_behavior(prompts, [_output("one", bad_response)], [_evaluation("one", 0.4)])
    result = compare_base_to_finetuned(base, finetuned)
    assert not result.passed
    assert any("repetition" in failure or "prompt_echo" in failure for failure in result.failures)


def test_gate_accepts_a_non_regressing_candidate() -> None:
    prompts = {"one": "Choose an architecture for a payment service."}
    response = "Use a modular monolith and revisit the choice as deployment ownership increases."
    base = summarize_behavior(prompts, [_output("one", response)], [_evaluation("one")])
    candidate = summarize_behavior(prompts, [_output("one", response)], [_evaluation("one", 0.81)])
    assert compare_base_to_finetuned(base, candidate).passed


def test_thinking_mode_cannot_use_greedy_benchmark_decoding(tmp_path: pytest.TempPathFactory) -> None:
    runner = BenchmarkRunner(
        dataset=BenchmarkDataset([]),
        config=InferenceConfig(enable_thinking=True, do_sample=False),
        results_dir=tmp_path / "results",  # type: ignore[operator]
    )
    with pytest.raises(ValueError, match="thinking-mode"):
        runner._init_real_model()


def test_adapter_identity_mismatch_is_rejected_before_peft_load(tmp_path: pytest.TempPathFactory) -> None:
    adapter = tmp_path / "checkpoint" / "model"  # type: ignore[operator]
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "wrong/base"}), encoding="utf-8"
    )
    (adapter.parent / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "base_model_identifier": "wrong/base",
                "base_revision": "main",
                "tokenizer_identifier": "Qwen/Qwen3-8B",
            }
        ),
        encoding="utf-8",
    )
    runner = BenchmarkRunner(
        dataset=BenchmarkDataset([]),
        config=InferenceConfig(adapter_path=str(adapter)),
        results_dir=tmp_path / "results",  # type: ignore[operator]
    )
    runner.model = object()
    with pytest.raises(ValueError, match="base model"):
        runner._load_verified_adapter()


def test_gate_releases_base_before_loading_4bit_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    configs: list[InferenceConfig] = []
    sample = BenchmarkSample("one", "test", "easy", "Small team payment system.", "Recommend an approach.")

    class FakeRunner:
        def __init__(self, dataset: BenchmarkDataset, config: InferenceConfig, *args: object, **kwargs: object) -> None:
            self.config = config
            configs.append(config)

        def run(self) -> tuple[object, list[EvaluationResult]]:
            events.append("run-adapter" if self.config.adapter_path else "run-base")
            return object(), [_evaluation("one")]

        def _load_completed_raw_outputs(self) -> dict[str, RawOutput]:
            return {
                "one": _output(
                    "one", "Use a modular monolith first and revisit service boundaries as ownership grows."
                )
            }

        def release_model(self) -> None:
            events.append("release-adapter" if self.config.adapter_path else "release-base")

    monkeypatch.setattr("architectai_pretraining.benchmark.runner.BenchmarkRunner", FakeRunner)
    result = run_base_vs_finetuned_gate(
        BenchmarkDataset([sample]), InferenceConfig(quantization="4bit"), "adapter", tmp_path
    )
    assert result.passed
    assert events == ["run-base", "release-base", "run-adapter", "release-adapter"]
    assert [config.quantization for config in configs] == ["4bit", "4bit"]
