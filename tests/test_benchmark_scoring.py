"""Unit tests for deterministic rubric scoring, unsupported claim detection, and human review export."""

import json

import pytest

from architectai_pretraining.benchmark.models import BenchmarkSample, RawOutput
from architectai_pretraining.benchmark.scoring import (
    DeterministicRubricEvaluator,
    export_human_review,
)


def test_deterministic_rubric_evaluator_label() -> None:
    evaluator = DeterministicRubricEvaluator()
    # User Correction #3: Evaluator must be explicitly labeled as proxy/heuristic
    assert evaluator.EVALUATOR_TYPE == "deterministic_proxy_v1"


def test_evaluator_unsupported_claim_penalty() -> None:
    evaluator = DeterministicRubricEvaluator()
    sample = BenchmarkSample(
        id="arch_001",
        category="microservices_vs_monolith",
        difficulty="medium",
        scenario="Evaluating architecture choices for payments service.",
        question="Should we use microservices?",
        rubric={},
    )

    # Response with absolute unsupported claims
    bad_response = "Microservices are always required for any application. Also NoSQL is always faster than SQL."
    raw_bad = RawOutput(
        sample_id="arch_001",
        model_identifier="Qwen/Qwen3-8B",
        prompt_hash="hash1",
        raw_response=bad_response,
        input_tokens=50,
        output_tokens=30,
        latency_seconds=0.5,
    )

    eval_bad = evaluator.evaluate(sample, raw_bad)
    assert len(eval_bad.unsupported_claims) >= 2
    assert eval_bad.scores["unsupported_claim_avoidance"] < 1.0
    assert eval_bad.evaluator_type == "deterministic_proxy_v1"

    # Good response avoiding absolute claims
    good_response = "A modular monolith is recommended given team constraints. However, if database CPU load exceeds 85% or p95 latency exceeds 500ms, revisit extracting the inventory service. Evaluate trade-offs between consistency and operational complexity."
    raw_good = RawOutput(
        sample_id="arch_001",
        model_identifier="Qwen/Qwen3-8B",
        prompt_hash="hash2",
        raw_response=good_response,
        input_tokens=50,
        output_tokens=40,
        latency_seconds=0.5,
    )

    eval_good = evaluator.evaluate(sample, raw_good)
    assert len(eval_good.unsupported_claims) == 0
    assert eval_good.scores["unsupported_claim_avoidance"] == 1.0
    assert len(eval_good.revisit_conditions_identified) >= 2


def test_export_human_review(tmp_path: pytest.TempPathFactory) -> None:
    sample = BenchmarkSample(
        id="arch_001",
        category="architecture_choice",
        difficulty="easy",
        scenario="Scenario description text.",
        question="Question?",
        facts=["Fact 1"],
        expected_considerations=["Consideration 1"],
        rubric={"weight": 1},
    )
    raw = RawOutput(
        sample_id="arch_001",
        model_identifier="Qwen/Qwen3-8B",
        prompt_hash="h1",
        raw_response="Base model response text",
        input_tokens=20,
        output_tokens=20,
        latency_seconds=0.1,
    )

    evaluator = DeterministicRubricEvaluator()
    eval_res = evaluator.evaluate(sample, raw)

    out_file = tmp_path / "human_review.jsonl"  # type: ignore[operator]
    export_human_review([sample], [raw], [eval_res], out_file)

    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    data = json.loads(lines[0])
    assert data["sample_id"] == "arch_001"
    assert data["proxy_evaluator"] == "deterministic_proxy_v1"
    assert "human_review_fields" in data
    assert data["human_review_fields"]["human_audited"] is False

# Benchmark test_benchmark_scoring.py test update
