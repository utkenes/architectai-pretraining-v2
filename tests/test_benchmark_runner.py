"""Unit tests for benchmark runner, resumable execution, and mock validation mode."""

import json

import pytest

from architectai_pretraining.benchmark.dataset import BenchmarkDataset
from architectai_pretraining.benchmark.models import BenchmarkSample, InferenceConfig
from architectai_pretraining.benchmark.runner import BenchmarkRunner


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

# Benchmark test_benchmark_runner.py test update
