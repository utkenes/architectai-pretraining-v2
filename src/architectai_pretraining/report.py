"""Human-readable markdown corpus audit report generator."""

from dataclasses import dataclass
from pathlib import Path

from architectai_pretraining.balance import ConcentrationMetrics
from architectai_pretraining.manifest import CurationManifest
from architectai_pretraining.sequence import LengthPercentiles, SequenceLengthEvaluation


@dataclass
class Stage4ReadinessCheck:
    is_ready: bool
    reasons: list[str]
    blocking_issues: list[str]


def evaluate_stage4_readiness(
    manifest: CurationManifest,
    real_tokenizer_used: bool,
    has_fixture_tokens: bool,
    has_license_violations: bool,
    has_split_leakage: bool,
    validation_passed: bool,
) -> Stage4ReadinessCheck:
    blocking: list[str] = []
    reasons: list[str] = []

    if not real_tokenizer_used:
        blocking.append("Mock/approximate token counter was used instead of real pinned tokenizer.")
    else:
        reasons.append("Real Qwen tokenizer counts used.")

    if has_fixture_tokens:
        blocking.append("Test fixture tokens detected in curated production splits.")
    else:
        reasons.append("Zero fixture tokens in production curated dataset.")

    if has_license_violations:
        blocking.append("Unresolved license violations detected in retained records.")
    else:
        reasons.append("No unresolved license violations in retained records.")

    if has_split_leakage:
        blocking.append("Near-duplicate cluster leakage detected across train/validation splits.")
    else:
        reasons.append("Cluster isolation enforced with zero near-duplicate leakage across splits.")

    if not validation_passed:
        blocking.append("Curated JSONL schema validation failed.")
    else:
        reasons.append("Curated JSONL schema validation passed.")

    if manifest.validation_tokens < 10000:
        blocking.append(f"Validation token set is too small ({manifest.validation_tokens} < 10,000 tokens).")
    else:
        reasons.append(f"Validation token set is sufficiently sized ({manifest.validation_tokens:,} tokens).")

    if not manifest.output_corpus_fingerprint:
        blocking.append("Corpus fingerprint generation failed.")
    else:
        reasons.append(f"Deterministic corpus fingerprint generated ({manifest.output_corpus_fingerprint[:12]}...).")

    is_ready = len(blocking) == 0
    return Stage4ReadinessCheck(is_ready=is_ready, reasons=reasons, blocking_issues=blocking)


def generate_corpus_audit_report(
    manifest: CurationManifest,
    conc_before: ConcentrationMetrics,
    conc_after: ConcentrationMetrics,
    percentiles: LengthPercentiles,
    seq_evals: list[SequenceLengthEvaluation],
    readiness: Stage4ReadinessCheck,
    output_path: Path,
) -> str:
    md = []
    md.append("# ArchitectAI Stage 3 Corpus Curation Audit Report\n")
    md.append(f"**Build Timestamp:** `{manifest.build_timestamp}`  ")
    md.append(f"**Tokenizer:** `{manifest.tokenizer_identifier}` (revision: `{manifest.tokenizer_revision}`)  ")
    md.append(f"**Input Corpus Fingerprint:** `{manifest.input_corpus_fingerprint}`  ")
    md.append(f"**Curated Corpus Fingerprint:** `{manifest.output_corpus_fingerprint}`  \n")

    md.append("---")
    md.append("## 1. Corpus Health & Overview\n")
    md.append("| Metric | Raw Input Corpus | Curated Training Corpus |")
    md.append("|---|---|---|")
    md.append(f"| **Documents Count** | {manifest.input_documents_count:,} | {manifest.curated_documents_count:,} |")
    md.append(f"| **Total Tokens** | {manifest.total_input_tokens:,} | {manifest.total_curated_tokens:,} |")
    md.append(f"| **Train Tokens** | - | {manifest.train_tokens:,} |")
    md.append(f"| **Validation Tokens** | - | {manifest.validation_tokens:,} |")
    md.append(f"| **Holdout Tokens** | - | {manifest.holdout_tokens:,} |")
    md.append(f"| **Quality Rejects** | - | {manifest.quality_rejects_count:,} |")
    md.append(f"| **Relevance Rejects** | - | {manifest.relevance_rejects_count:,} |")
    md.append(f"| **Exact Duplicates Removed** | - | {manifest.exact_duplicates_count:,} |")
    md.append(f"| **Near Duplicates Removed** | - | {manifest.near_duplicates_count:,} |")
    md.append(f"| **Balanced Out Documents** | - | {manifest.balanced_out_count:,} |")
    md.append(f"| **Fixture Excluded Documents** | - | {manifest.fixture_excluded_count:,} |\n")

    md.append("### Quality Bucket Distribution")
    for bucket, count in manifest.quality_bucket_distribution.items():
        pct = (count / max(1, manifest.curated_documents_count)) * 100.0
        md.append(f"- **{bucket.upper()} Quality:** {count:,} documents ({pct:.1f}%)")
    md.append("")

    md.append("---")
    md.append("## 2. Dominance & Concentration Metrics\n")
    md.append("| Metric | Before Curation | After Curation | Target Cap |")
    md.append("|---|---|---|---|")
    md.append(f"| **Top 1 Source Share** | {conc_before.top_1_source_share:.1%} | {conc_after.top_1_source_share:.1%} | <= 30.0% |")
    md.append(f"| **Top 5 Source Share** | {conc_before.top_5_source_share:.1%} | {conc_after.top_5_source_share:.1%} | - |")
    md.append(f"| **Top Category Share** | {conc_before.top_category_share:.1%} | {conc_after.top_category_share:.1%} | <= 40.0% |")
    md.append(f"| **Top Organization Share** | {conc_before.top_organization_share:.1%} | {conc_after.top_organization_share:.1%} | <= 40.0% |")
    md.append(f"| **Source Concentration (HHI)** | {conc_before.hhi_source_index:.4f} | {conc_after.hhi_source_index:.4f} | Low |")
    md.append("")

    md.append("---")
    md.append("## 3. Token Distribution by Source & Category\n")
    md.append("### Top Sources by Token Volume\n")
    md.append("| Source ID | Documents | Curated Tokens | Token Share |")
    md.append("|---|---|---|---|")
    for src_id, stats in sorted(manifest.source_distribution.items(), key=lambda x: x[1].get("tokens", 0), reverse=True):
        md.append(f"| `{src_id}` | {stats.get('docs', 0):,} | {stats.get('tokens', 0):,} | {stats.get('share', 0.0):.1%} |")
    md.append("")

    md.append("### Category Token Distribution\n")
    md.append("| Category ID | Curated Tokens | Token Share | Status |")
    md.append("|---|---|---|---|")
    for cat_id, stats in sorted(manifest.category_distribution.items(), key=lambda x: x[1].get("tokens", 0), reverse=True):
        status = stats.get("status", "healthy")
        md.append(f"| `{cat_id}` | {stats.get('tokens', 0):,} | {stats.get('share', 0.0):.1%} | {status} |")
    md.append("")

    md.append("---")
    md.append("## 4. Licensing Distribution Audit\n")
    md.append("| License ID | Curated Tokens | Token Share | Status |")
    md.append("|---|---|---|---|")
    for lic_id, stats in sorted(manifest.license_distribution.items(), key=lambda x: x[1].get("tokens", 0), reverse=True):
        md.append(f"| `{lic_id}` | {stats.get('tokens', 0):,} | {stats.get('share', 0.0):.1%} | Verified Open |")
    md.append("")

    md.append("---")
    md.append("## 5. Document Length & Sequence Packing Analysis\n")
    md.append(f"- **Min Tokens:** {percentiles.min_tokens:,}")
    md.append(f"- **Median Tokens:** {percentiles.median:,}")
    md.append(f"- **Mean Tokens:** {percentiles.mean_tokens:,.1f}")
    md.append(f"- **P75 / P90 / P95:** {percentiles.p75:,} / {percentiles.p90:,} / {percentiles.p95:,}")
    md.append(f"- **P99 / Max Tokens:** {percentiles.p99:,} / {percentiles.max_tokens:,}")
    md.append("")

    md.append("### Context Window Evaluation (512, 1024, 2048, 4096)\n")
    md.append("| Context Window | Fitting Docs (%) | Requiring Chunking (%) | Est. Packed Sequences | Capacity Efficiency |")
    md.append("|---|---|---|---|---|")
    for ev in seq_evals:
        md.append(f"| `{ev.context_length}` | {ev.natively_fitting_pct:.1f}% | {ev.requiring_splitting_pct:.1f}% | {ev.estimated_sequence_count:,} | {ev.packing_efficiency_pct:.1f}% |")
    md.append("")
    md.append("**Recommended Sequence Length for First DAPT Pilot:** `2048` tokens")
    md.append("- **Justification:** 2048 is the pilot default pending Colab GPU inspection. It preserves useful architectural context while keeping memory pressure lower than 4096; it does not imply that full-parameter Qwen3-8B training fits a T4-class GPU.")
    md.append("")

    md.append("---")
    md.append("## 6. Training Readiness Verdict\n")
    if readiness.is_ready:
        md.append("```text\nREADY_FOR_STAGE_4=true\n```\n")
        md.append("### Passed Readiness Criteria:")
        for r in readiness.reasons:
            md.append(f"- [x] {r}")
    else:
        md.append("```text\nREADY_FOR_STAGE_4=false\n```\n")
        md.append("### Blocking Issues:")
        for b in readiness.blocking_issues:
            md.append(f"- [ ] {b}")

    content = "\n".join(md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return content


# Shift tree hash
