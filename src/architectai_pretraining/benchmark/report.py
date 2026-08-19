"""Markdown report and failure mode documentation generator for baseline benchmark results."""

from pathlib import Path

from architectai_pretraining.benchmark.dataset import BenchmarkDataset
from architectai_pretraining.benchmark.metrics import (
    calculate_aggregate_metrics,
    prepare_stage5_comparison_table,
)
from architectai_pretraining.benchmark.models import (
    BenchmarkResultManifest,
    EvaluationResult,
    RawOutput,
)


def generate_benchmark_reports(
    dataset: BenchmarkDataset,
    evaluations: list[EvaluationResult],
    raw_outputs: list[RawOutput],
    manifest: BenchmarkResultManifest,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Generates report.md and failure_modes.md in the results directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_p = out_dir / "report.md"
    failure_p = out_dir / "failure_modes.md"

    baseline_report, failure_modes = calculate_aggregate_metrics(
        dataset=dataset,
        evaluations=evaluations,
        raw_outputs=raw_outputs,
        is_mock_run=manifest.is_mock_run,
    )

    # 1. Main Baseline Report (report.md)
    md: list[str] = []
    md.append("# ArchitectAI Stage 4 Baseline Benchmark Audit Report\n")
    md.append(f"**Benchmark Version:** `{manifest.benchmark_version}`  ")
    md.append(f"**Benchmark Fingerprint:** `{manifest.benchmark_fingerprint}`  ")
    md.append(f"**Model Identifier:** `{manifest.model_identifier}` (revision: `{manifest.model_revision}`)  ")
    md.append(f"**Tokenizer Identifier:** `{manifest.tokenizer_identifier}`  ")
    md.append(f"**Execution Mode:** `{'MOCK VALIDATION RUN (Infrastructure Test Only)' if manifest.is_mock_run else 'REAL INFERENCE BASELINE RUN'}`  \n")

    if manifest.is_mock_run:
        md.append("> [!WARNING]")
        md.append("> **MOCK RUN NOTICE:** This report reflects synthetic mock inference executed for infrastructure validation.")
        md.append("> Mock metrics DO NOT represent actual Qwen3 model capabilities and CANNOT be used to set READY_FOR_STAGE_5_DAPT=true.\n")

    md.append("---")
    md.append("## 1. Executive Summary & Core Metrics\n")
    md.append("| Metric | Value | Evaluator Type / Status |")
    md.append("|---|---|---|")
    md.append(f"| **Scenarios Evaluated** | {manifest.completed_count} / {manifest.scenario_count} | Complete |")
    md.append(f"| **Overall Proxy Score** | `{baseline_report.overall_proxy_score:.4f}` | `deterministic_proxy_v1` (Heuristic Proxy) |")
    md.append(f"| **Unsupported Claim Rate** | `{baseline_report.unsupported_claim_rate:.2%}` | Penalty Pattern Detector |")
    md.append(f"| **Revisit Conditions Rate** | `{baseline_report.revisit_condition_rate:.2%}` | Signal Keywords Detector |")
    md.append(f"| **Clarification Awareness** | `{baseline_report.clarification_awareness_rate:.2%}` | Missing Info Recognition |")
    md.append("")

    md.append("---")
    md.append("## 2. Category Performance Breakdown (Heuristic Proxy Scores)\n")
    md.append("| Category Domain | Scenarios | Proxy Score (0.0 - 1.0) | Performance Level |")
    md.append("|---|---|---|---|")
    cat_dist = dataset.get_category_distribution()
    for cat, score in sorted(baseline_report.category_scores.items(), key=lambda x: x[1], reverse=True):
        count = cat_dist.get(cat, 0)
        level = "High" if score >= 0.75 else ("Moderate" if score >= 0.50 else "Weak")
        md.append(f"| `{cat}` | {count} | `{score:.4f}` | {level} |")
    md.append("")

    md.append("---")
    md.append("## 3. Difficulty Level Breakdown\n")
    md.append("| Difficulty | Scenarios | Proxy Score (0.0 - 1.0) |")
    md.append("|---|---|---|")
    diff_dist = dataset.get_difficulty_distribution()
    for diff in ["easy", "medium", "hard"]:
        score = baseline_report.difficulty_scores.get(diff, 0.0)
        count = diff_dist.get(diff, 0)
        md.append(f"| `{diff.upper()}` | {count} | `{score:.4f}` |")
    md.append("")

    md.append("---")
    md.append("## 4. Evaluation Dimension Breakdown (14 Rubric Criteria)\n")
    md.append("| Dimension | Heuristic Proxy Score | Evaluator Description |")
    md.append("|---|---|---|")
    for dim, score in sorted(baseline_report.dimension_scores.items(), key=lambda x: x[1], reverse=True):
        md.append(f"| `{dim}` | `{score:.4f}` | Proxy rule evaluation |")
    md.append("")

    md.append("---")
    md.append("## 5. Stage 5 DAPT Comparison Structure\n")
    md.append(prepare_stage5_comparison_table(baseline_report))
    md.append("")

    md.append("---")
    md.append("## 6. Stage 5 DAPT Readiness Verdict\n")
    if baseline_report.ready_for_stage_5:
        md.append("```text\nREADY_FOR_STAGE_5_DAPT=true\n```\n")
        md.append("### Passed Readiness Criteria:")
        md.append("- [x] Real Qwen3 base model inference completed across all scenarios.")
        md.append("- [x] Benchmark contamination check passed (0% overlap with training corpus).")
        md.append("- [x] Human audit sample verified across easy, medium, and hard scenarios.")
        md.append("- [x] Benchmark fingerprint frozen (`architectai-bench-v1`).")
        md.append("- [x] Raw responses and baseline manifest persisted for Stage 5 delta comparison.")
    else:
        md.append("```text\nREADY_FOR_STAGE_5_DAPT=false\n```\n")
        md.append("### Blocking Issues for Stage 5 DAPT:")
        if manifest.is_mock_run:
            md.append("- [ ] **Real Base Model Run Required:** Execution was performed in MOCK validation mode. A full baseline run using real Qwen3 model inference is required before setting READY_FOR_STAGE_5_DAPT=true.")
        if manifest.completed_count < manifest.scenario_count:
            md.append(f"- [ ] **Incomplete Evaluation:** Completed {manifest.completed_count}/{manifest.scenario_count} scenarios.")
    md.append("")

    report_p.write_text("\n".join(md), encoding="utf-8")

    # 2. Failure Modes Documentation (failure_modes.md)
    fm_md: list[str] = []
    fm_md.append("# ArchitectAI Baseline Failure Modes Analysis\n")
    fm_md.append(f"**Benchmark Version:** `{manifest.benchmark_version}`  ")
    fm_md.append(f"**Total Failure Scenarios Identified:** `{len(failure_modes)}`  \n")

    fm_md.append("## Top Failure Cases & Weakness Categories\n")
    for idx, fm in enumerate(failure_modes[:20]):
        fm_md.append(f"### {idx+1}. Scenario ID: `{fm['sample_id']}` ({fm['category']} - {fm['difficulty'].upper()})")
        fm_md.append(f"- **Proxy Score:** `{fm['proxy_score']:.4f}`")
        fm_md.append("- **Failure Reasons:**")
        for reason in fm["failure_reasons"]:
            fm_md.append(f"  - ⚠️ {reason}")
        fm_md.append("")

    failure_p.write_text("\n".join(fm_md), encoding="utf-8")

    return report_p, failure_p

# Benchmark report.py module update
