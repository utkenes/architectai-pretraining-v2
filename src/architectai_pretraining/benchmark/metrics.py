"""Aggregate metrics and Stage 5 comparison structures for benchmark reporting."""

from typing import Any

from architectai_pretraining.benchmark.dataset import BenchmarkDataset
from architectai_pretraining.benchmark.models import BaselineReport, EvaluationResult, RawOutput


def calculate_aggregate_metrics(
    dataset: BenchmarkDataset,
    evaluations: list[EvaluationResult],
    raw_outputs: list[RawOutput],
    is_mock_run: bool = False,
) -> tuple[BaselineReport, list[dict[str, Any]]]:
    """Calculates aggregate metrics, category/difficulty breakdowns, and failure modes."""
    if not evaluations:
        return (
            BaselineReport(
                overall_proxy_score=0.0,
                category_scores={},
                difficulty_scores={},
                dimension_scores={},
                unsupported_claim_rate=0.0,
                revisit_condition_rate=0.0,
                clarification_awareness_rate=0.0,
                total_samples=0,
                is_mock_run=is_mock_run,
                ready_for_stage_5=False,
            ),
            [],
        )

    total = len(evaluations)
    overall_score = round(sum(e.overall_proxy_score for e in evaluations) / total, 4)

    # Category Breakdown
    cat_totals: dict[str, list[float]] = {}
    diff_totals: dict[str, list[float]] = {}
    dim_totals: dict[str, list[float]] = {}

    unsupported_claim_counts = 0
    revisit_condition_counts = 0
    clarification_ack_counts = 0

    for ev in evaluations:
        # Category
        if ev.category not in cat_totals:
            cat_totals[ev.category] = []
        cat_totals[ev.category].append(ev.overall_proxy_score)

        # Difficulty
        if ev.difficulty not in diff_totals:
            diff_totals[ev.difficulty] = []
        diff_totals[ev.difficulty].append(ev.overall_proxy_score)

        # Dimension scores
        for dim, val in ev.scores.items():
            if dim not in dim_totals:
                dim_totals[dim] = []
            dim_totals[dim].append(val)

        if ev.unsupported_claims:
            unsupported_claim_counts += 1

        if ev.revisit_conditions_identified:
            revisit_condition_counts += 1

        if ev.missing_info_acknowledged:
            clarification_ack_counts += 1

    cat_scores = {
        cat: round(sum(vals) / len(vals), 4) for cat, vals in cat_totals.items()
    }
    diff_scores = {
        diff: round(sum(vals) / len(vals), 4) for diff, vals in diff_totals.items()
    }
    dim_scores = {
        dim: round(sum(vals) / len(vals), 4) for dim, vals in dim_totals.items()
    }

    unsupported_rate = round(unsupported_claim_counts / total, 4)
    revisit_rate = round(revisit_condition_counts / total, 4)
    clarification_rate = round(clarification_ack_counts / total, 4)

    # Identify Failure Modes
    failure_modes: list[dict[str, Any]] = []
    samples_map = {s.id: s for s in dataset.samples}

    for ev in sorted(evaluations, key=lambda x: x.overall_proxy_score):
        s = samples_map.get(ev.sample_id)
        if not s:
            continue

        reasons: list[str] = []
        if ev.unsupported_claims:
            reasons.append(f"Made unsupported claims: {', '.join(ev.unsupported_claims)}")
        if ev.scores.get("tradeoff_quality", 0) < 0.4:
            reasons.append("Weak trade-off reasoning")
        if ev.scores.get("reliability_reasoning", 0) < 0.4:
            reasons.append("Missing reliability/resilience mechanisms")
        if ev.scores.get("revisit_conditions", 0) < 0.4:
            reasons.append("Lacked quantitative revisit condition signals")
        if s.metadata.get("has_missing_info") and not ev.missing_info_acknowledged:
            reasons.append("Failed to identify missing critical information")

        if reasons or ev.overall_proxy_score < 0.5:
            failure_modes.append(
                {
                    "sample_id": ev.sample_id,
                    "category": ev.category,
                    "difficulty": ev.difficulty,
                    "proxy_score": ev.overall_proxy_score,
                    "failure_reasons": reasons if reasons else ["Low overall proxy score"],
                }
            )

    ready_for_stage_5 = not is_mock_run and total == len(dataset.samples)

    report = BaselineReport(
        overall_proxy_score=overall_score,
        category_scores=cat_scores,
        difficulty_scores=diff_scores,
        dimension_scores=dim_scores,
        unsupported_claim_rate=unsupported_rate,
        revisit_condition_rate=revisit_rate,
        clarification_awareness_rate=clarification_rate,
        total_samples=total,
        is_mock_run=is_mock_run,
        ready_for_stage_5=ready_for_stage_5,
    )

    return report, failure_modes


def prepare_stage5_comparison_table(baseline_report: BaselineReport) -> str:
    """Formats markdown table structure for Stage 5 DAPT comparison without fabricating Stage 5 data."""
    md = []
    md.append("| Metric / Dimension | Untrained Base Qwen3 (Stage 4) | ArchitectAI DAPT (Stage 5) | Delta |")
    md.append("|---|---|---|---|")
    md.append(f"| **Overall Proxy Score** | `{baseline_report.overall_proxy_score:.4f}` | *TBD (Stage 5)* | *TBD* |")
    md.append(f"| **Unsupported Claim Avoidance** | `{baseline_report.dimension_scores.get('unsupported_claim_avoidance', 0.0):.4f}` | *TBD (Stage 5)* | *TBD* |")
    md.append(f"| **Trade-off Quality** | `{baseline_report.dimension_scores.get('tradeoff_quality', 0.0):.4f}` | *TBD (Stage 5)* | *TBD* |")
    md.append(f"| **Reliability Reasoning** | `{baseline_report.dimension_scores.get('reliability_reasoning', 0.0):.4f}` | *TBD (Stage 5)* | *TBD* |")
    md.append(f"| **Scalability Reasoning** | `{baseline_report.dimension_scores.get('scalability_reasoning', 0.0):.4f}` | *TBD (Stage 5)* | *TBD* |")
    md.append(f"| **Revisit Conditions Rate** | `{baseline_report.revisit_condition_rate:.4f}` | *TBD (Stage 5)* | *TBD* |")
    md.append(f"| **Clarification Awareness** | `{baseline_report.clarification_awareness_rate:.4f}` | *TBD (Stage 5)* | *TBD* |")

    return "\n".join(md)

# Benchmark metrics.py module update
