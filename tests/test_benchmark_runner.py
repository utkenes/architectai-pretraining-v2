"""Unit tests for benchmark runner, resumable execution, and mock validation mode."""

import hashlib
import json

import pytest

from architectai_pretraining.benchmark.dataset import BenchmarkDataset
from architectai_pretraining.benchmark.models import BenchmarkSample, InferenceConfig
from architectai_pretraining.benchmark.runner import BenchmarkRunner


def test_inference_config_default_completion_budget() -> None:
    assert InferenceConfig().max_new_tokens == 2048
    assert InferenceConfig().system_prompt_profile == "frozen"


def test_runner_mock_execution(tmp_path: pytest.TempPathFactory) -> None:
    results_dir = tmp_path / "results"  # type: ignore[operator]

    sample1 = BenchmarkSample(
        id="arch_001",
        category="microservices_vs_monolith",
        difficulty="easy",
        scenario="A startup building a payment MVP with 5 engineers.",
        question="Monolith or microservices?",
        facts=["5 engineers", "MVP payment"],
        rubric={"test": 1.0},
    )

    dataset = BenchmarkDataset([sample1])
    config = InferenceConfig(model_identifier="Qwen/Qwen3-8B")

    runner = BenchmarkRunner(
        dataset=dataset,
        config=config,
        results_dir=results_dir,
        use_mock=True,
    )

    manifest, evaluations = runner.run()

    assert manifest.scenario_count == 1
    assert manifest.completed_count == 1
    assert manifest.is_mock_run
    # User Correction #1: NEVER set ready_for_stage_5=True from mock runs!
    assert not manifest.ready_for_stage_5

    assert (results_dir / "raw_outputs.jsonl").exists()
    assert (results_dir / "evaluations.jsonl").exists()
    assert (results_dir / "human_review.jsonl").exists()
    assert (results_dir / "baseline_manifest.json").exists()

    with open(results_dir / "baseline_manifest.json", encoding="utf-8") as f:
        data = json.load(f)
        assert data["is_mock_run"] is True
        assert data["ready_for_stage_5"] is False

    raw_output = json.loads((results_dir / "raw_outputs.jsonl").read_text(encoding="utf-8"))
    assert raw_output["ended_with_eos"] is None
    assert raw_output["hit_max_new_tokens"] is False
    assert raw_output["stop_reason"] == "mock"


def test_runner_resumable_execution(tmp_path: pytest.TempPathFactory) -> None:
    results_dir = tmp_path / "results_resume"  # type: ignore[operator]

    sample1 = BenchmarkSample(
        id="arch_001",
        category="microservices_vs_monolith",
        difficulty="easy",
        scenario="Scenario 1 text content.",
        question="Question 1?",
        rubric={"test": 1.0},
    )
    sample2 = BenchmarkSample(
        id="arch_002",
        category="messaging",
        difficulty="medium",
        scenario="Scenario 2 text content.",
        question="Question 2?",
        rubric={"test": 1.0},
    )

    dataset_1 = BenchmarkDataset([sample1])
    runner1 = BenchmarkRunner(dataset=dataset_1, results_dir=results_dir, use_mock=True)
    runner1.run()

    # Verify 1 output completed
    completed_1 = runner1._load_completed_raw_outputs()
    assert "arch_001" in completed_1
    assert "arch_002" not in completed_1

    # Second run with 2 samples should resume and skip arch_001
    dataset_2 = BenchmarkDataset([sample1, sample2])
    runner2 = BenchmarkRunner(dataset=dataset_2, results_dir=results_dir, use_mock=True)
    manifest, evaluations = runner2.run()

    assert manifest.completed_count == 2
    completed_2 = runner2._load_completed_raw_outputs()
    assert "arch_001" in completed_2
    assert "arch_002" in completed_2


def test_frozen_profile_keeps_legacy_result_fingerprint(tmp_path: pytest.TempPathFactory) -> None:
    dataset = BenchmarkDataset([])
    config = InferenceConfig()
    legacy_config = config.to_dict()
    legacy_config.pop("system_prompt_profile")
    expected = hashlib.sha256(
        json.dumps(legacy_config | {"adapter_contents": ""}, sort_keys=True).encode("utf-8")
    ).hexdigest()

    runner = BenchmarkRunner(dataset, config=config, results_dir=tmp_path / "results", use_mock=True)  # type: ignore[operator]
    assert runner._inference_fingerprint == expected


def test_legacy_raw_output_without_completion_telemetry_remains_readable(
    tmp_path: pytest.TempPathFactory,
) -> None:
    results_dir = tmp_path / "legacy_results"  # type: ignore[operator]
    runner = BenchmarkRunner(BenchmarkDataset([]), results_dir=results_dir, use_mock=True)
    legacy_raw = {
        "sample_id": "arch_legacy",
        "model_identifier": "Qwen/Qwen3-8B",
        "prompt_hash": "hash",
        "raw_response": "A complete response.",
        "input_tokens": 5,
        "output_tokens": 4,
        "latency_seconds": 0.1,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "is_mock": True,
        "inference_fingerprint": runner._inference_fingerprint,
    }
    (results_dir / "raw_outputs.jsonl").write_text(json.dumps(legacy_raw) + "\n", encoding="utf-8")

    loaded = runner._load_completed_raw_outputs()["arch_legacy"]
    assert loaded.ended_with_eos is None
    assert loaded.hit_max_new_tokens is False
    assert loaded.stop_reason == "unknown"

# Benchmark test_benchmark_runner.py test update


def test_v2_1_mock_execution_writes_decision_quality_telemetry(
    tmp_path: pytest.TempPathFactory,
) -> None:
    sample = BenchmarkSample(
        id="arch_v21",
        category="architecture_choice",
        difficulty="easy",
        scenario="A service has 8 engineers and a stated throughput requirement.",
        question="Choose a direction.",
        facts=["8 engineers", "stated throughput requirement"],
    )
    results_dir = tmp_path / "v21_results"  # type: ignore[operator]
    runner = BenchmarkRunner(
        BenchmarkDataset([sample]),
        config=InferenceConfig(system_prompt_profile="fact_bounded_decision_v2_1"),
        results_dir=results_dir,
        use_mock=True,
    )

    runner.run()

    records = [json.loads(line) for line in (results_dir / "decision_quality.jsonl").read_text().splitlines()]
    summary = json.loads((results_dir / "decision_quality_summary.json").read_text())
    assert records[0]["static_check_status"] == "supported"
    assert records[0]["requires_human_review"] is True
    assert summary["evaluator_type"] == "decision_quality_static_v2_1"
    assert "proxy_score" not in summary
