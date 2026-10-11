"""GPU-free tests for V2.1 evidence and decision validation."""

import json

from architectai_pretraining.benchmark.decision_quality import (
    decision_quality_summary,
    evaluate_decision_quality,
    validate_decision_submission,
)
from architectai_pretraining.benchmark.models import BenchmarkSample, RawOutput


def _sample() -> BenchmarkSample:
    return BenchmarkSample(
        id="synthetic_status_case",
        category="architecture_choice",
        difficulty="medium",
        scenario=(
            "The service must sustain 50000 RPS. The team has 8 engineers. "
            "The monthly budget is $10000."
        ),
        question="Choose an architecture direction.",
        facts=["50000 RPS", "8 engineers", "$10000 monthly budget"],
    )


def _response(
    status: str,
    facts: list[dict[str, str]],
    material_unknowns: list[dict[str, str]] | None = None,
    conflicts: list[dict[str, str]] | None = None,
    possible: bool = True,
    recommendation: str = "Use a capability-level direction supported by the scenario.",
) -> str:
    return json.dumps(
        {
            "evidence_check": {
                "explicit_facts": facts,
                "material_unknowns": material_unknowns or [],
                "nonmaterial_unknowns": [],
                "conflicts": conflicts or [],
                "provisional_direction": {
                    "possible": possible,
                    "direction": "Use the least complex capability direction that remains safe.",
                },
            },
            "decision": {"status": status, "recommendation": recommendation},
        }
    )


def test_recommended_with_two_grounded_facts_is_only_statically_supported() -> None:
    record = validate_decision_submission(
        _sample(),
        _response(
            "RECOMMENDED",
            [
                {"claim": "Throughput is required.", "scenario_evidence": "50000 RPS"},
                {"claim": "Team capacity is given.", "scenario_evidence": "8 engineers"},
            ],
        ),
    )

    assert record.static_check_status == "supported"
    assert record.grounded_fact_count == 2
    assert record.requires_human_review is True
    assert "recommendation_sufficiency_requires_human_review" in record.semantic_review_flags


def test_conditional_requires_a_material_unknown_and_provisional_direction() -> None:
    record = validate_decision_submission(
        _sample(),
        _response(
            "CONDITIONAL",
            [{"claim": "Throughput is required.", "scenario_evidence": "50000 RPS"}],
            material_unknowns=[
                {
                    "driver": "delivery semantics",
                    "why_material": "It can change persistence and acknowledgement design.",
                }
            ],
        ),
    )

    assert record.static_check_status == "supported"
    assert record.declared_status == "CONDITIONAL"
    assert "unknown_materiality_requires_human_review" in record.semantic_review_flags


def test_blocked_requires_two_grounded_sides_of_a_conflict() -> None:
    sample = BenchmarkSample(
        id="conflict_case",
        category="architecture_choice",
        difficulty="hard",
        scenario="Data must remain only in region A. Data must remain only in region B.",
        question="Choose a storage direction.",
    )
    record = validate_decision_submission(
        sample,
        _response(
            "BLOCKED",
            [],
            conflicts=[
                {
                    "description": "Exclusive residency constraints conflict.",
                    "left_evidence": "Data must remain only in region A.",
                    "right_evidence": "Data must remain only in region B.",
                }
            ],
            possible=False,
        ),
    )

    assert record.static_check_status == "supported"
    assert record.declared_status == "BLOCKED"
    assert "conflict_resolvability_requires_human_review" in record.semantic_review_flags


def test_invalid_evidence_and_unsupported_number_never_auto_pass() -> None:
    record = validate_decision_submission(
        _sample(),
        _response(
            "RECOMMENDED",
            [{"claim": "Invented fact.", "scenario_evidence": "99.99% availability"}],
            material_unknowns=[
                {"driver": "ordering", "why_material": "It changes the design."}
            ],
            recommendation="Use Kafka with a 99.99% target.",
        ),
    )

    assert record.static_check_status == "invalid"
    assert record.requires_human_review is True
    assert any("not a verbatim quote" in issue for issue in record.issues)
    assert any("Unsupported numeric values" in issue for issue in record.issues)
    assert "kafka" in record.unsupported_named_terms


def test_incomplete_schema_is_unavailable_and_requires_review() -> None:
    payload = {
        "evidence_check": {
            "explicit_facts": [],
            "nonmaterial_unknowns": [],
            "conflicts": [],
            "provisional_direction": {"possible": True, "direction": "A provisional direction."},
        },
        "decision": {"status": "CONDITIONAL", "recommendation": "Wait for a material driver."},
    }
    record = validate_decision_submission(_sample(), json.dumps(payload))

    assert record.static_check_status == "unavailable"
    assert record.requires_human_review is True
    assert any("material_unknowns" in issue for issue in record.issues)

def test_named_term_detection_uses_word_boundaries() -> None:
    record = validate_decision_submission(
        _sample(),
        _response(
            "RECOMMENDED",
            [
                {"claim": "Throughput is required.", "scenario_evidence": "50000 RPS"},
                {"claim": "Team capacity is given.", "scenario_evidence": "8 engineers"},
            ],
            recommendation="Always choose the least-complex direction supported by the facts.",
        ),
    )

    assert "aws" not in record.unsupported_named_terms

def test_malformed_evidence_is_unavailable_and_requires_review() -> None:
    record = validate_decision_submission(_sample(), "RECOMMENDED: use a monolith.")

    assert record.static_check_status == "unavailable"
    assert record.requires_human_review is True
    assert record.declared_status is None


def test_decision_quality_metrics_are_independent_of_proxy_scores() -> None:
    sample = _sample()
    valid = _response(
        "RECOMMENDED",
        [
            {"claim": "Throughput is required.", "scenario_evidence": "50000 RPS"},
            {"claim": "Team capacity is given.", "scenario_evidence": "8 engineers"},
        ],
    )
    raw = RawOutput("synthetic_status_case", "test", "hash", valid, 1, 1, 0.0)
    records, summary = evaluate_decision_quality([sample], [raw])

    assert len(records) == 1
    assert summary["evaluator_type"] == "decision_quality_static_v2_1"
    assert summary["static_supported_rate"] == 1.0
    assert "proxy_score" not in summary
    assert decision_quality_summary(records) == summary
