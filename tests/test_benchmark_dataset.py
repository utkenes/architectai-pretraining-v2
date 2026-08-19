"""Unit tests for benchmark dataset loading, validation, and fingerprinting."""

from pathlib import Path

from architectai_pretraining.benchmark.dataset import (
    BenchmarkDataset,
    compute_benchmark_fingerprint,
    load_benchmark_dataset,
)
from architectai_pretraining.benchmark.models import BenchmarkSample


def test_benchmark_sample_creation_and_dict() -> None:
    sample = BenchmarkSample(
        id="test_arch_001",
        category="architecture_choice",
        difficulty="medium",
        scenario="A high throughput API needing scalable architecture.",
        question="Which pattern should be used?",
        facts=["10,000 RPS", "5 engineers"],
        expected_considerations=["Modularity", "Operational simplicity"],
        rubric={"driver_weight": 0.3},
    )

    d = sample.to_dict()
    assert d["id"] == "test_arch_001"
    assert d["difficulty"] == "medium"

    recreated = BenchmarkSample.from_dict(d)
    assert recreated.id == sample.id
    assert recreated.category == sample.category


def test_benchmark_dataset_validation() -> None:
    s1 = BenchmarkSample(
        id="sample_1",
        category="distributed_systems",
        difficulty="easy",
        scenario="Valid scenario description exceeding twenty characters length.",
        question="Valid question?",
        rubric={"test": 1},
    )
    dataset = BenchmarkDataset([s1])
    is_valid, errors = dataset.validate()
    assert is_valid
    assert len(errors) == 0

    # Invalid sample (short scenario & invalid difficulty)
    s_invalid = BenchmarkSample(
        id="sample_invalid",
        category="invalid_category",
        difficulty="super_hard",
        scenario="Too short",
        question="Q?",
        rubric={},
    )
    dataset_invalid = BenchmarkDataset([s_invalid])
    is_valid_inv, errors_inv = dataset_invalid.validate()
    assert not is_valid_inv
    assert len(errors_inv) >= 2


def test_benchmark_fingerprint_determinism() -> None:
    s1 = BenchmarkSample(
        id="arch_1",
        category="messaging",
        difficulty="medium",
        scenario="Message queue scaling scenario with 10k messages/sec.",
        question="How to scale consumer throughput?",
        rubric={"test": 1.0},
    )
    s2 = BenchmarkSample(
        id="arch_2",
        category="reliability",
        difficulty="hard",
        scenario="Circuit breaker and retry policy under 500 server errors.",
        question="What failure isolation pattern applies?",
        rubric={"test": 2.0},
    )

    fp1 = compute_benchmark_fingerprint([s1, s2])
    fp2 = compute_benchmark_fingerprint([s2, s1])  # Order invariance

    assert fp1 == fp2
    assert len(fp1) == 64


def test_human_audit_sample_selection() -> None:
    samples = []
    for i in range(10):
        samples.append(BenchmarkSample(id=f"e_{i}", category="cat", difficulty="easy", scenario="Scenario content text long enough.", question="Question text?", rubric={"a": 1}))
    for i in range(15):
        samples.append(BenchmarkSample(id=f"m_{i}", category="cat", difficulty="medium", scenario="Scenario content text long enough.", question="Question text?", rubric={"a": 1}))
    for i in range(15):
        samples.append(BenchmarkSample(id=f"h_{i}", category="cat", difficulty="hard", scenario="Scenario content text long enough.", question="Question text?", rubric={"a": 1}))

    dataset = BenchmarkDataset(samples)
    audit = dataset.get_audit_sample(min_easy=5, min_medium=10, min_hard=10)

    assert len(audit) == 25
    easy_count = sum(1 for s in audit if s.difficulty == "easy")
    med_count = sum(1 for s in audit if s.difficulty == "medium")
    hard_count = sum(1 for s in audit if s.difficulty == "hard")

    assert easy_count == 5
    assert med_count == 10
    assert hard_count == 10


def test_load_real_benchmark_v1_file() -> None:
    file_path = Path("data/benchmark/architectai_v1.jsonl")
    if file_path.exists():
        dataset = load_benchmark_dataset(file_path)
        assert len(dataset.samples) >= 80
        assert dataset.fingerprint is not None
        is_valid, errors = dataset.validate()
        assert is_valid, f"Validation errors: {errors}"

# Benchmark test_benchmark_dataset.py test update
