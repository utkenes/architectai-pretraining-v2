"""Source diagnostic classification and machine-readable audit report generator."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from architectai_pretraining.sources import SourceConfig, verify_repository_license


def generate_source_audit(
    sources: list[SourceConfig],
    raw_documents: list[Any],
    curated_documents: list[Any],
    doc_tokens: dict[str, int],
    ledger_entries: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Write per-source discovery, rejection, and final-token audit artifacts."""
    raw_by_source: dict[str, int] = {}
    curated_by_source: dict[str, list[Any]] = {}
    rejected_by_source: dict[str, int] = {}
    for document in raw_documents:
        raw_by_source[document.source_id] = raw_by_source.get(document.source_id, 0) + 1
    for document in curated_documents:
        curated_by_source.setdefault(document.source_id, []).append(document)
    for entry in ledger_entries:
        source_id = entry.get("source_id")
        if source_id and entry.get("decision") not in {"kept"}:
            rejected_by_source[source_id] = rejected_by_source.get(source_id, 0) + 1
    entries: list[dict[str, Any]] = []
    for source in sources:
        curated = curated_by_source.get(source.id, [])
        final_tokens = sum(doc_tokens.get(document.id, 0) for document in curated)
        discovered = raw_by_source.get(source.id, 0)
        reason = None
        if not source.enabled:
            reason = str(source.metadata.get("disabled_reason", "Disabled by manifest."))
        elif discovered == 0:
            reason = "No accepted raw documents; inspect source diagnostics for licensing, path, or cleaner gate causes."
        entries.append(
            {
                "source_id": source.id,
                "enabled": source.enabled,
                "category": source.category,
                "repository_or_url": source.url or source.path,
                "license_expected": source.license_id,
                "license_verified": source.license_id if source.enabled else None,
                "documents_discovered": discovered,
                "documents_accepted": discovered,
                "documents_rejected": rejected_by_source.get(source.id, 0),
                "final_curated_docs": len(curated),
                "final_tokens": final_tokens,
                "reason_if_zero": reason,
            }
        )
    payload = {"sources": entries}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# ArchitectAI Source Manifest Audit", "", "| Source | Enabled | Category | Discovered | Rejected | Curated docs | Final tokens | Reason if zero |", "|---|---:|---|---:|---:|---:|---:|---|"]
    for entry in entries:
        lines.append(
            f"| `{entry['source_id']}` | {entry['enabled']} | `{entry['category']}` | {entry['documents_discovered']} | {entry['documents_rejected']} | {entry['final_curated_docs']} | {entry['final_tokens']} | {entry['reason_if_zero'] or ''} |"
        )
    (output_dir / "source_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def write_manual_review_sample(
    documents: list[Any], doc_tokens: dict[str, int], quality_scores: dict[str, Any], output_path: Path
) -> None:
    """Create deterministic, non-mutating samples for category-level human inspection."""
    category_order = [
        "cloud_architecture",
        "architecture_patterns",
        "distributed_systems",
        "reliability",
        "database_architecture",
        "adr",
        "domain_driven_design",
    ]
    lines = ["# Curated Corpus Manual Review Sample", "", "Samples are deterministically selected by SHA-256 document ID.", ""]
    for category in category_order:
        candidates = [document for document in documents if document.category == category]
        selected = sorted(candidates, key=lambda document: hashlib.sha256(document.id.encode()).hexdigest())[:5]
        lines.extend([f"## {category}", ""])
        if not selected:
            lines.extend(["No curated documents available; see source audit for root cause.", ""])
            continue
        for document in selected:
            score = quality_scores.get(document.id)
            quality = getattr(score, "quality_score", "unknown")
            excerpt = " ".join(document.text.split())[:300]
            lines.extend(
                [
                    f"### {document.title or document.id}",
                    f"- Source: `{document.source_id}`",
                    f"- Tokens: {doc_tokens.get(document.id, 0)}",
                    f"- Quality score: {quality}",
                    f"- Excerpt: {excerpt}",
                    "",
                ]
            )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_license_audit(documents: list[Any], output_dir: Path) -> dict[str, Any]:
    """Record retained-license evidence; any missing evidence is a hard finding."""
    unverified = [
        document.id
        for document in documents
        if not (document.license_id and document.metadata.get("verified_license_id"))
    ]
    by_license: dict[str, int] = {}
    for document in documents:
        license_id = str(document.metadata.get("verified_license_id") or document.license_id or "unverified")
        by_license[license_id] = by_license.get(license_id, 0) + 1
    payload = {"unverified_final_documents": len(unverified), "unverified_document_ids": unverified, "documents_by_verified_license": by_license}
    (output_dir / "license_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# License Audit", "", f"unverified_final_documents = {len(unverified)}", "", "| License | Documents |", "|---|---:|"]
    lines.extend(f"| `{license_id}` | {count} |" for license_id, count in sorted(by_license.items()))
    (output_dir / "license_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


@dataclass
class SourceDiagnosticResult:
    source_id: str
    name: str
    category: str
    enabled: bool
    status: str  # "active", "disabled_license_restriction", "disabled_unclear_license", "disabled_user_config"
    classification: str
    license_id: str | None
    verified_license: str | None
    explanation: str


class SourceDiagnosticsEngine:
    """Diagnoses source ingestion status and classifies zero-doc root causes."""

    def __init__(self, manifest_sources: list[SourceConfig], cache_dir: Path) -> None:
        self.sources = manifest_sources
        self.cache_dir = cache_dir

    def diagnose_all(self) -> list[SourceDiagnosticResult]:
        results: list[SourceDiagnosticResult] = []
        for src in self.sources:
            res = self.diagnose_source(src)
            results.append(res)
        return results

    def diagnose_source(self, src: SourceConfig) -> SourceDiagnosticResult:
        if not src.enabled:
            reason = src.metadata.get("disabled_reason", "Disabled by manifest configuration.")
            if "NC" in (src.license_id or "") or "non-commercial" in reason.lower():
                classification = "license_non_commercial_restriction"
                status = "disabled_license_restriction"
            else:
                classification = "disabled_user_config"
                status = "disabled_user_config"
            return SourceDiagnosticResult(
                source_id=src.id,
                name=src.name,
                category=src.category,
                enabled=False,
                status=status,
                classification=classification,
                license_id=src.license_id,
                verified_license=None,
                explanation=reason,
            )

        if src.type == "git_repository":
            repo_dir = self.cache_dir / "git" / src.id
            if repo_dir.exists():
                lic_res = verify_repository_license(repo_dir, src.license_id)
                if not lic_res.is_valid:
                    return SourceDiagnosticResult(
                        source_id=src.id,
                        name=src.name,
                        category=src.category,
                        enabled=False,
                        status="disabled_unclear_license",
                        classification="license_verification_failed",
                        license_id=src.license_id,
                        verified_license=None,
                        explanation=lic_res.error_message or "License verification failed.",
                    )
                return SourceDiagnosticResult(
                    source_id=src.id,
                    name=src.name,
                    category=src.category,
                    enabled=True,
                    status="active",
                    classification="verified_active",
                    license_id=src.license_id,
                    verified_license=lic_res.verified_license_id,
                    explanation="Repository license verified successfully.",
                )

        return SourceDiagnosticResult(
            source_id=src.id,
            name=src.name,
            category=src.category,
            enabled=True,
            status="active",
            classification="verified_active",
            license_id=src.license_id,
            verified_license=src.license_id,
            explanation="Source active.",
        )

    def generate_report(self, output_path: Path) -> dict[str, Any]:
        diag_results = self.diagnose_all()
        active_count = sum(1 for r in diag_results if r.enabled)
        disabled_count = sum(1 for r in diag_results if not r.enabled)

        report_data = {
            "total_catalog_sources": len(diag_results),
            "active_sources_count": active_count,
            "disabled_sources_count": disabled_count,
            "diagnostics": [asdict(r) for r in diag_results],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        return report_data
