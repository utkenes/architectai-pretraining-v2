"""Audit-first preparation of the external architecture corpus.

This module deliberately stops at frozen DAPT text records.  It never creates
chat/SFT examples and it never reads the benchmark as an input source.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from architectai_pretraining.cleaner import BoilerplateCleaner, TextCleaner
from architectai_pretraining.code_prose import CodeProseAnalyzer
from architectai_pretraining.dedup import ExactDeduplicator
from architectai_pretraining.io import write_dict_jsonl, write_jsonl
from architectai_pretraining.manifest import compute_corpus_fingerprint
from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.near_dedup import MinHashLSHDeduplicator
from architectai_pretraining.relevance import ArchitectureRelevanceScorer, DomainRelevanceGate
from architectai_pretraining.scoring import DocumentQualityScorer
from architectai_pretraining.semantic import (
    annotate_document,
    category_coverage_report,
    coverage_report,
    group_adjacent_sections,
    related_for_grouping,
)
from architectai_pretraining.sources import (
    SourceConfig,
    _resolve_configured_path,
    get_adapter,
    load_source_manifest,
    verify_repository_license,
)
from architectai_pretraining.splitter import GroupCorpusSplitter
from architectai_pretraining.tokenizer import (
    CachingTokenCounter,
    HuggingFaceTokenCounter,
    TokenCounter,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CorpusV2Config:
    config_path: Path
    corpus_version: str
    target_tokens: int
    token_tolerance: float
    seed: int
    tokenizer_identifier: str
    tokenizer_revision: str
    max_section_tokens: int
    min_relevance_score: float
    max_link_ratio: float
    max_code_ratio: float
    borderline_relevance_score: float
    borderline_quality_score: float
    rejection_sample_size: int
    max_source_share: float
    category_tolerance: float
    concept_aware_selection: bool
    allow_preview_backfill: bool
    allow_freeze_backfill: bool
    category_targets: dict[str, float]
    concept_min_tokens: int
    concept_min_sources: int
    concept_min_documents: int
    max_concept_dominant_source_share: float
    split_ratios: tuple[float, float, float]
    source_configs: list[SourceConfig]


def load_corpus_v2_config(path: str | Path) -> CorpusV2Config:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    tokenizer = data.get("tokenizer", {})
    quality = data.get("quality", {})
    balancing = data.get("balancing", {})
    coverage = data.get("coverage", {})
    split = data.get("split", {})
    targets = {
        key: float(value) for key, value in (balancing.get("category_token_targets") or {}).items()
    }
    if not targets:
        raise ValueError("corpus_v2 requires balancing.category_token_targets")
    if abs(sum(targets.values()) - 1.0) > 0.001:
        raise ValueError("Category token targets must sum to 1.0.")
    return CorpusV2Config(
        config_path=config_path,
        corpus_version=str(data["corpus_version"]),
        target_tokens=int(data.get("target_tokens", 1_000_000)),
        token_tolerance=float(data.get("token_tolerance", 0.05)),
        seed=int(data.get("seed", 42)),
        tokenizer_identifier=str(tokenizer.get("identifier", "Qwen/Qwen3-8B")),
        tokenizer_revision=str(tokenizer.get("revision", "main")),
        max_section_tokens=int(data.get("max_section_tokens", 2400)),
        min_relevance_score=float(quality.get("min_architecture_relevance_score", 0.40)),
        max_link_ratio=float(quality.get("max_link_ratio", 0.22)),
        max_code_ratio=float(quality.get("max_code_ratio", 0.55)),
        borderline_relevance_score=float(quality.get("borderline_relevance_score", 0.28)),
        borderline_quality_score=float(quality.get("borderline_quality_score", 0.33)),
        rejection_sample_size=int(quality.get("rejection_sample_size", 24)),
        max_source_share=float(balancing.get("max_source_token_share", 0.15)),
        category_tolerance=float(balancing.get("category_tolerance", 0.05)),
        concept_aware_selection=bool(balancing.get("concept_aware_selection", True)),
        allow_preview_backfill=bool(balancing.get("allow_preview_backfill", True)),
        allow_freeze_backfill=bool(balancing.get("allow_freeze_backfill", False)),
        category_targets=targets,
        concept_min_tokens=int(coverage.get("min_concept_tokens", 1_000)),
        concept_min_sources=int(coverage.get("min_concept_sources", 2)),
        concept_min_documents=int(coverage.get("min_concept_documents", 2)),
        max_concept_dominant_source_share=float(coverage.get("max_dominant_source_share", 0.70)),
        split_ratios=(
            float(split.get("train_ratio", 0.90)),
            float(split.get("validation_ratio", 0.05)),
            float(split.get("heldout_ratio", 0.05)),
        ),
        source_configs=load_source_manifest(config_path),
    )


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_structural_headings(doc: CorpusDocument) -> CorpusDocument:
    """Normalize reStructuredText underline headings for common section logic.

    This preserves the source prose while exposing chapter boundaries to the
    semantic sectionizer, quality scorer, and configured section exclusions.
    It is intentionally structural normalization, not a relevance threshold
    change: non-prose and link-heavy sections still go through the same gates.
    """
    text = re.sub(
        r"(?m)^(?P<title>[^\n]+)\n(?P<underline>[=\-`:'\"~^_*+#<>]{3,})\s*$",
        lambda match: f"# {match.group('title').strip()}",
        doc.text,
    )
    return doc.model_copy(update={"text": text}) if text != doc.text else doc


def _license_concerns(status: str) -> list[str]:
    return {
        "approved": [],
        "mixed": ["mixed content/license classes"],
        "restrictive": ["restrictive terms"],
        "unverified": ["no verified reusable-content license"],
        "noncommercial": ["non-commercial restriction"],
        "custom_terms": ["custom/restrictive author terms"],
    }.get(status, ["manual license review required"])


def _distribution(docs: list[CorpusDocument]) -> dict[str, dict[str, dict[str, float | int]]]:
    total = max(1, sum(doc.token_count or 0 for doc in docs))
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for key in ("sources", "categories"):
        counts: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"documents": 0, "tokens": 0}
        )
        for doc in docs:
            value = doc.source_id if key == "sources" else (doc.primary_category or doc.category)
            counts[value]["documents"] = int(counts[value]["documents"]) + 1
            counts[value]["tokens"] = int(counts[value]["tokens"]) + int(doc.token_count or 0)
        result[key] = {
            name: {**stats, "share": round(int(stats["tokens"]) / total, 4)}
            for name, stats in sorted(counts.items())
        }
    return result


def _sectionize(
    doc: CorpusDocument, counter: TokenCounter, max_tokens: int
) -> list[CorpusDocument]:
    """Split only at semantic headings, then paragraphs if one section is huge."""
    group_id = str(doc.metadata.get("provenance_group_id") or doc.id)
    text = doc.text
    if counter.count(text) <= max_tokens:
        headings = doc.section_headings or ([doc.title] if doc.title else [])
        return [
            doc.model_copy(
                update={
                    "section_headings": headings,
                    "metadata": {
                        **doc.metadata,
                        "provenance_group_id": group_id,
                        "section_index": 0,
                    },
                }
            )
        ]
    parts = re.split(r"(?m)(?=^(?:#{1,6}\s+|={1,6}\s+))", text)
    sections: list[tuple[str | None, str]] = []
    current_title: str | None = None
    for part in parts:
        if not part.strip():
            continue
        first = part.splitlines()[0].strip()
        if re.match(r"^(#{1,6}\s+|={1,6}\s+)", first):
            current_title = re.sub(r"^(#{1,6}\s+|={1,6}\s+)", "", first).strip()
        sections.append((current_title, part.strip()))
    if not sections:
        sections = [(doc.title, text)]
    output: list[CorpusDocument] = []
    for section_title, section in sections:
        paragraphs = re.split(r"\n{2,}", section)
        chunk: list[str] = []
        for paragraph in paragraphs:
            prospective = "\n\n".join(chunk + [paragraph]).strip()
            if chunk and counter.count(prospective) > max_tokens:
                output.append(
                    _derived_section(doc, group_id, section_title, "\n\n".join(chunk), len(output))
                )
                chunk = [paragraph]
            else:
                chunk.append(paragraph)
        if chunk:
            output.append(
                _derived_section(doc, group_id, section_title, "\n\n".join(chunk), len(output))
            )
    return output


def _derived_section(
    doc: CorpusDocument, group_id: str, section_title: str | None, text: str, index: int
) -> CorpusDocument:
    digest = _content_hash(f"{doc.id}:{index}:{text}")[:16]
    return doc.model_copy(
        update={
            "id": digest,
            "text": text,
            "section_title": section_title,
            "section_headings": [section_title] if section_title else [],
            "metadata": {**doc.metadata, "provenance_group_id": group_id, "section_index": index},
        }
    )


def _strip_policy_sections(doc: CorpusDocument, patterns: list[str] | None) -> CorpusDocument:
    """Remove explicitly configured low-value Markdown/AsciiDoc sections before scoring."""
    if not patterns:
        return doc
    parts = re.split(r"(?m)(?=^(?:#{1,6}\s+|={1,6}\s+))", doc.text)
    kept: list[str] = []
    lowered = tuple(pattern.lower() for pattern in patterns)
    for part in parts:
        first = part.splitlines()[0].lower() if part.splitlines() else ""
        if any(pattern in first for pattern in lowered):
            continue
        kept.append(part)
    text = "".join(kept).strip()
    return doc.model_copy(update={"text": text}) if text else doc


def _assign_section_category(doc: CorpusDocument, rules: dict[str, str]) -> CorpusDocument:
    title = (doc.section_title or doc.title or "").lower()
    for pattern, category in rules.items():
        if pattern.lower() in title:
            # Legacy source rules are section-level priors in v3, never a
            # source-authoritative final category.
            return doc.model_copy(
                update={"metadata": {**doc.metadata, "section_category_hint": category}}
            )
    return doc


@dataclass
class SectionAssessment:
    """One deterministic section decision before optional same-document rescue."""

    document: CorpusDocument
    relevance_score: float
    quality_score: float
    code_ratio: float
    link_ratio: float
    decision: str
    reason: str


def _score_histogram(values: list[float], boundaries: tuple[float, float, float, float]) -> dict[str, int]:
    """Stable score buckets for capacity diagnostics, without per-unit dumps."""
    first, second, third, fourth = boundaries
    labels = (
        f"<{first:.2f}",
        f"{first:.2f}-{second:.2f}",
        f"{second:.2f}-{third:.2f}",
        f"{third:.2f}-{fourth:.2f}",
        f">={fourth:.2f}",
    )
    counts = dict.fromkeys(labels, 0)
    for value in values:
        label = (
            labels[0]
            if value < first
            else labels[1]
            if value < second
            else labels[2]
            if value < third
            else labels[3]
            if value < fourth
            else labels[4]
        )
        counts[label] += 1
    return counts


def _rejection_samples(
    assessments: list[SectionAssessment], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    """Return a bounded, deterministic audit sample without changing corpus output."""
    groups = {
        "normal_pass": "strong_accepted",
        "rescued_borderline": "rescued",
        "borderline_rejected": "borderline_rejected",
        "hard_rejected": "hard_rejected",
    }
    per_group = max(1, sample_size // len(groups))
    samples: list[dict[str, Any]] = []
    for decision, sample_class in groups.items():
        matching = sorted(
            (assessment for assessment in assessments if assessment.decision == decision),
            key=lambda assessment: hashlib.sha256(
                f"{seed}:{assessment.document.id}".encode()
            ).hexdigest(),
        )[:per_group]
        for assessment in matching:
            doc = assessment.document
            samples.append(
                {
                    "sample_class": sample_class,
                    "unit_id": doc.id,
                    "source_id": doc.source_id,
                    "relative_path": doc.relative_path,
                    "document_title": doc.title,
                    "section_title": doc.section_title,
                    "token_count": doc.token_count,
                    "relevance_score": assessment.relevance_score,
                    "quality_score": assessment.quality_score,
                    "code_ratio": assessment.code_ratio,
                    "primary_category": doc.primary_category,
                    "related_concepts": doc.related_concepts,
                    "decision": assessment.decision,
                    "reason": assessment.reason,
                    "text": doc.text,
                }
            )
    return samples


def _section_index(doc: CorpusDocument) -> int:
    return int(doc.metadata.get("section_index", 0))


def _same_provenance(left: CorpusDocument, right: CorpusDocument) -> bool:
    left_group = left.metadata.get("provenance_group_id")
    right_group = right.metadata.get("provenance_group_id")
    return (
        left.source_id == right.source_id
        and left.relative_path == right.relative_path
        and (left_group is None or right_group is None or left_group == right_group)
    )


def _enrich_scored_section(
    doc: CorpusDocument,
    *,
    relevance: ArchitectureRelevanceScorer,
    quality: DocumentQualityScorer,
    code: CodeProseAnalyzer,
    domain: DomainRelevanceGate,
    config: CorpusV2Config,
    source: SourceConfig,
    counter: TokenCounter,
) -> SectionAssessment:
    """Apply hard gates first, preserving a narrow auditable rescue pool."""
    domain_result = domain.check(doc)
    relevance_score = relevance.score(doc)
    quality_result = quality.score(doc)
    code_result = code.analyze(doc, counter)
    enriched = doc.model_copy(
        update={
            "token_count": code_result.total_tokens,
            "content_sha256": _content_hash(doc.text),
            "quality_score": quality_result.quality_score,
            "architecture_relevance_score": relevance_score.score,
            "code_ratio": code_result.code_to_prose_ratio,
            "corpus_version": config.corpus_version,
            "source_priority": source.source_priority,
            "source_name": doc.source_name or source.name,
            "relative_path": doc.relative_path or str(doc.metadata.get("relative_path", "")),
            "verified_license_id": doc.verified_license_id or doc.metadata.get("verified_license_id"),
        }
    )
    hard_reasons: list[str] = []
    if doc.primary_category is None:
        hard_reasons.append("unresolved_classification")
    if not domain_result.is_relevant:
        hard_reasons.append(domain_result.reason or "domain_relevance")
    if relevance_score.link_ratio > config.max_link_ratio:
        hard_reasons.append("link_ratio_exceeds_configured_maximum")
    if code_result.is_code_dominated:
        hard_reasons.append("code_ratio_exceeds_configured_threshold")
    if hard_reasons:
        return SectionAssessment(
            enriched,
            relevance_score.score,
            quality_result.quality_score,
            code_result.code_to_prose_ratio,
            relevance_score.link_ratio,
            "hard_rejected",
            "; ".join(hard_reasons),
        )
    if relevance_score.passed and quality_result.quality_score >= quality.min_document_score:
        return SectionAssessment(
            enriched,
            relevance_score.score,
            quality_result.quality_score,
            code_result.code_to_prose_ratio,
            relevance_score.link_ratio,
            "normal_pass",
            "; ".join(relevance_score.reasons) or "normal quality and relevance acceptance",
        )
    if (
        source.license_training_status == "approved"
        and relevance_score.score >= config.borderline_relevance_score
        and quality_result.quality_score >= config.borderline_quality_score
    ):
        return SectionAssessment(
            enriched,
            relevance_score.score,
            quality_result.quality_score,
            code_result.code_to_prose_ratio,
            relevance_score.link_ratio,
            "borderline",
            "eligible for same-document contextual rescue",
        )
    return SectionAssessment(
        enriched,
        relevance_score.score,
        quality_result.quality_score,
        code_result.code_to_prose_ratio,
        relevance_score.link_ratio,
        "borderline_rejected",
        "below primary threshold and outside the conservative contextual rescue window",
    )




class CorpusV2Pipeline:
    """Deterministic audit, preview, and explicit freeze workflow."""

    def __init__(
        self,
        config_path: str | Path = "configs/corpus_v2.yaml",
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.config = load_corpus_v2_config(config_path)
        self._counter = CachingTokenCounter(token_counter) if token_counter else None

    @property
    def counter(self) -> CachingTokenCounter:
        """Load the real tokenizer only when text preparation actually starts."""
        if self._counter is None:
            self._counter = CachingTokenCounter(
                HuggingFaceTokenCounter(
                    self.config.tokenizer_identifier,
                    self.config.tokenizer_revision,
                    fallback_allowed_in_prod=False,
                )
            )
        return self._counter

    def _evaluate_candidates(
        self, candidates: list[CorpusDocument], policies: dict[str, SourceConfig]
    ) -> tuple[list[CorpusDocument], list[SectionAssessment]]:
        """Keep primary gates strict, then attempt bounded same-document rescue."""
        relevance = ArchitectureRelevanceScorer(
            self.config.min_relevance_score, self.config.max_link_ratio
        )
        quality = DocumentQualityScorer(min_document_score=0.45)
        code = CodeProseAnalyzer(self.config.max_code_ratio)
        domain = DomainRelevanceGate()
        assessments = [
            _enrich_scored_section(
                doc,
                relevance=relevance,
                quality=quality,
                code=code,
                domain=domain,
                config=self.config,
                source=policies[doc.source_id],
                counter=self.counter,
            )
            for doc in candidates
        ]
        eligible = [
            assessment
            for assessment in assessments
            if assessment.decision in {"normal_pass", "borderline"}
        ]
        ordered = sorted(
            eligible,
            key=lambda item: (
                item.document.source_id,
                item.document.relative_path or "",
                _section_index(item.document),
                item.document.id,
            ),
        )
        consumed: set[str] = set()
        rescued: list[CorpusDocument] = []
        for assessment in ordered:
            if assessment.decision != "borderline" or assessment.document.id in consumed:
                continue
            neighbours = [
                candidate
                for candidate in ordered
                if candidate.document.id not in consumed
                and candidate.document.id != assessment.document.id
                and abs(_section_index(candidate.document) - _section_index(assessment.document)) == 1
                and _same_provenance(assessment.document, candidate.document)
                and related_for_grouping(assessment.document, candidate.document)
            ]
            neighbours.sort(key=lambda item: (_section_index(item.document), item.document.id))
            for neighbour in neighbours:
                pair = sorted(
                    [assessment.document, neighbour.document], key=lambda doc: (_section_index(doc), doc.id)
                )
                combined = group_adjacent_sections(pair, self.counter, self.config.max_section_tokens)
                if len(combined) != 1:
                    continue
                rescored = _enrich_scored_section(
                    combined[0],
                    relevance=relevance,
                    quality=quality,
                    code=code,
                    domain=domain,
                    config=self.config,
                    source=policies[combined[0].source_id],
                    counter=self.counter,
                )
                if rescored.decision != "normal_pass":
                    continue
                rescued.append(
                    rescored.document.model_copy(
                        update={
                            "metadata": {
                                **rescored.document.metadata,
                                "recall_decision": "rescued_borderline",
                                "rescue_reason": "same-document adjacent semantic context",
                            }
                        }
                    )
                )
                assessment.decision = "rescued_borderline"
                assessment.reason = "rescued with adjacent same-document semantic section"
                if neighbour.decision == "borderline":
                    neighbour.decision = "rescued_borderline"
                    neighbour.reason = "rescued with adjacent same-document semantic section"
                consumed.update({assessment.document.id, neighbour.document.id})
                break
            if assessment.document.id not in consumed:
                assessment.decision = "borderline_rejected"
                assessment.reason = "no compatible adjacent same-document semantic section passed rescue"
        accepted = [
            assessment.document
            for assessment in assessments
            if assessment.decision == "normal_pass" and assessment.document.id not in consumed
        ]
        accepted.extend(rescued)
        return accepted, assessments

    def inventory(self) -> dict[str, Any]:
        sources: list[dict[str, Any]] = []
        for source in self.config.source_configs:
            root = _resolve_configured_path(source.path) if source.path else None
            files = get_adapter(source).list_candidate_files() if source.enabled else []
            raw_bytes = sum(path.stat().st_size for path in files)
            verification = (
                verify_repository_license(root, source.license_id)
                if root and root.is_dir()
                else None
            )
            sources.append(
                {
                    "source_id": source.id,
                    "enabled": source.enabled,
                    "category": source.category,
                    "root": str(root) if root else None,
                    "candidate_files": len(files),
                    "raw_bytes": raw_bytes,
                    "estimated_tokens": raw_bytes // 4,
                    "license_id": source.license_id,
                    "license_verified": verification.is_valid if verification else False,
                    "license_issue": verification.error_message
                    if verification and not verification.is_valid
                    else None,
                    "license_policy_type": source.license_policy.get("mode", "repository_wide"),
                    "license_evidence_path": source.license_evidence_path,
                    "license_training_status": source.license_training_status,
                    "license_review_status": source.license_review_status,
                    "release_eligible": source.release_eligible,
                    "commercial_reuse_permitted": source.commercial_reuse_permitted,
                    "license_concerns": _license_concerns(source.license_training_status),
                    "useful_prose_paths": source.include_patterns or [],
                    "rejected_paths": source.exclude_patterns or [],
                    "parser": source.parser,
                    "notes": source.notes,
                }
            )
        return {"corpus_version": self.config.corpus_version, "sources": sources}

    def write_inventory(self, output_dir: str | Path) -> dict[str, Any]:
        result = self.inventory()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "inventory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def license_audit(self) -> dict[str, Any]:
        inventory = self.inventory()
        return {
            "corpus_version": self.config.corpus_version,
            "source_reviews": inventory["sources"],
            "release_eligible_sources": [
                source["source_id"] for source in inventory["sources"] if source["release_eligible"]
            ],
            "release_ineligible_sources": [
                source["source_id"]
                for source in inventory["sources"]
                if not source["release_eligible"]
            ],
        }

    def write_license_audit(self, output_dir: str | Path) -> dict[str, Any]:
        result = self.license_audit()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "license_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def capacity(self) -> dict[str, Any]:
        """Calculate candidate and post-filter capacity without selecting a corpus."""
        policies = {source.id: source for source in self.config.source_configs}
        raw_docs: list[CorpusDocument] = []
        for source in self.config.source_configs:
            raw_docs.extend(get_adapter(source).ingest())
        cleaner = TextCleaner()
        boilerplate = BoilerplateCleaner()
        candidates: list[CorpusDocument] = []
        for raw in raw_docs:
            source = policies[raw.source_id]
            cleaned = _normalize_structural_headings(
                boilerplate.clean_document(cleaner.clean_document(raw))
            )
            cleaned = _strip_policy_sections(cleaned, source.strip_section_patterns)
            candidates.extend(
                annotate_document(_assign_section_category(section, source.section_category_rules))
                for section in _sectionize(cleaned, self.counter, self.config.max_section_tokens)
            )
        accepted, assessments = self._evaluate_candidates(candidates, policies)
        rejected_by_source: Counter[str] = Counter()
        rejection_reasons: dict[str, Counter[str]] = defaultdict(Counter)
        accepted_documents: dict[str, set[str]] = defaultdict(set)
        rejected_documents: dict[str, set[str]] = defaultdict(set)

        def document_key(doc: CorpusDocument) -> str:
            """Stable source-document identity before sections are grouped."""
            return str(doc.metadata.get("provenance_group_id") or doc.relative_path or doc.id)

        for assessment in assessments:
            doc = assessment.document
            if assessment.decision in {"normal_pass", "rescued_borderline"}:
                accepted_documents[doc.source_id].add(document_key(doc))
                continue
            rejected_by_source[doc.source_id] += 1
            rejection_reasons[doc.source_id].update([assessment.reason])
            rejected_documents[doc.source_id].add(document_key(doc))
        quality_passing_units = sum(
            assessment.decision == "normal_pass" for assessment in assessments
        )
        grouped = group_adjacent_sections(accepted, self.counter, self.config.max_section_tokens)
        grouped = [
            doc.model_copy(
                update={
                    "token_count": self.counter.count(doc.text),
                    "content_sha256": _content_hash(doc.text),
                }
            )
            for doc in grouped
        ]
        normal_tokens = sum(
            doc.token_count or 0
            for doc in accepted
            if doc.metadata.get("recall_decision") != "rescued_borderline"
        )
        rescued_tokens = sum(
            doc.token_count or 0
            for doc in accepted
            if doc.metadata.get("recall_decision") == "rescued_borderline"
        )
        exact = ExactDeduplicator().deduplicate(grouped)
        quality_rank = {
            doc.id: float(doc.token_count or 0) / 1_000_000 for doc in exact.deduplicated_documents
        }
        near = MinHashLSHDeduplicator(similarity_threshold=0.85).deduplicate(
            exact.deduplicated_documents, quality_rank
        )
        edition_sources = {"seven_edition", "book_edition"}
        edition_docs = [doc for doc in grouped if doc.source_id in edition_sources]
        hashes: dict[str, set[str]] = defaultdict(set)
        for doc in edition_docs:
            hashes[doc.content_sha256 or ""].add(doc.source_id)
        cross_exact = sum(1 for source_ids in hashes.values() if source_ids == edition_sources)
        near_doc_map = {doc.id: doc for doc in exact.deduplicated_documents}
        cross_near = [
            cluster
            for cluster in near.clusters
            if {
                near_doc_map[doc_id].source_id
                for doc_id in [cluster.canonical_document_id, *cluster.removed_document_ids]
            }
            == edition_sources
        ]
        edition_saved = sum(
            near_doc_map[doc_id].token_count or 0
            for cluster in cross_near
            for doc_id in cluster.removed_document_ids
        )
        eligible = near.canonical_documents
        source_tokens: Counter[str] = Counter()
        category_tokens: Counter[str] = Counter()
        source_docs: Counter[str] = Counter()
        for doc in eligible:
            source_tokens[doc.source_id] += doc.token_count or 0
            category_tokens[doc.primary_category or doc.category] += doc.token_count or 0
            source_docs[doc.source_id] += 1
        capped_source_tokens = {
            source.id: min(
                source.source_token_cap or self.config.target_tokens, source_tokens[source.id]
            )
            for source in self.config.source_configs
        }
        maximum = sum(capped_source_tokens.values())
        category_coverage: dict[str, dict[str, Any]] = {}
        for category, share in self.config.category_targets.items():
            available = category_tokens[category]
            desired = int(self.config.target_tokens * share)
            contributors = sorted(
                {doc.source_id for doc in eligible if (doc.primary_category or doc.category) == category}
            )
            cause = "available"
            if not contributors:
                cause = "no_source_configured"
            elif not available:
                cause = "quality_rejection_or_insufficient_raw_content"
            elif available < desired:
                cause = "insufficient_eligible_content_or_source_cap"
            category_coverage[category] = {
                "target_share": share,
                "target_tokens": desired,
                "available_eligible_tokens": available,
                "deficit_tokens": max(0, desired - available),
                "deficit_percentage": round(max(0, desired - available) / max(1, desired), 4),
                "contributing_sources": contributors,
                "cause": cause,
            }
        release_docs = [doc for doc in eligible if bool(doc.metadata.get("release_eligible"))]
        semantic = coverage_report(
            eligible,
            min_tokens=self.config.concept_min_tokens,
            min_sources=self.config.concept_min_sources,
            min_documents=self.config.concept_min_documents,
            max_dominant_source_share=self.config.max_concept_dominant_source_share,
        )
        source_report = {
            source.id: {
                "candidate_documents": sum(1 for doc in raw_docs if doc.source_id == source.id),
                "source_documents_discovered": sum(1 for doc in raw_docs if doc.source_id == source.id),
                "source_documents_with_any_accepted_unit": len(accepted_documents[source.id]),
                "source_documents_with_any_rejected_unit": len(rejected_documents[source.id]),
                "documents_discovered": sum(1 for doc in raw_docs if doc.source_id == source.id),
                "documents_accepted": len(accepted_documents[source.id]),
                "documents_rejected": len(rejected_documents[source.id] - accepted_documents[source.id]),
                "documents_with_rejected_sections": len(rejected_documents[source.id]),
                "documents_with_quality_passing_units": len(accepted_documents[source.id]),
                "eligible_documents": source_docs[source.id],
                "quality_passing_units_before_grouping": sum(
                    1 for doc in accepted if doc.source_id == source.id
                ),
                "training_units": source_docs[source.id],
                "rejected_documents": rejected_by_source[source.id],
                "legacy_rejected_documents_note": "Deprecated: this is a section/unit count, not a document count.",
                "quality_rejected_units": rejected_by_source[source.id],
                "sections_normal_pass": sum(
                    assessment.decision == "normal_pass" and assessment.document.source_id == source.id
                    for assessment in assessments
                ),
                "sections_rescued": sum(
                    assessment.decision == "rescued_borderline" and assessment.document.source_id == source.id
                    for assessment in assessments
                ),
                "sections_rejected": rejected_by_source[source.id],
                "major_rejection_reasons": dict(rejection_reasons[source.id].most_common()),
                "eligible_tokens": source_tokens[source.id],
                "tokens": source_tokens[source.id],
                "capped_tokens": capped_source_tokens[source.id],
                "license_training_status": source.license_training_status,
                "release_eligible": source.release_eligible,
                "license_review_status": source.license_review_status,
                "license_evidence_path": source.license_evidence_path,
            }
            for source in self.config.source_configs
        }
        return {
            "target_tokens": self.config.target_tokens,
            "funnel": {
                "documents_discovered": len(raw_docs),
                "sections_generated": len(candidates),
                "sections_normal_pass": sum(assessment.decision == "normal_pass" for assessment in assessments),
                "sections_borderline": sum(
                    assessment.decision in {"rescued_borderline", "borderline_rejected"}
                    for assessment in assessments
                ),
                "sections_rescued": sum(
                    assessment.decision == "rescued_borderline" for assessment in assessments
                ),
                "sections_rejected": sum(
                    assessment.decision in {"hard_rejected", "borderline_rejected"}
                    for assessment in assessments
                ),
                "tokens_normal_pass": normal_tokens,
                "tokens_rescued": rescued_tokens,
                "tokens_rejected": sum(
                    assessment.document.token_count or 0
                    for assessment in assessments
                    if assessment.decision in {"hard_rejected", "borderline_rejected"}
                ),
                "quality_passing_units_before_grouping": quality_passing_units,
                "units_after_same_document_grouping": len(grouped),
                "units_after_dedup": len(eligible),
            },
            "theoretical_source_cap_capacity": sum(
                source.source_token_cap or self.config.target_tokens
                for source in self.config.source_configs
            ),
            "estimated_eligible_token_capacity": sum(source_tokens.values()),
            "expected_maximum_achievable_tokens": maximum,
            "release_eligible_capacity": sum(doc.token_count or 0 for doc in release_docs),
            "source_capacity": source_report,
            "category_capacity": category_coverage,
            "missing_categories": [
                key for key, value in category_coverage.items() if value["deficit_tokens"]
            ],
            "exact_duplicates_removed": exact.duplicates_removed,
            "near_duplicates_removed": len(near.removed_documents),
            "cross_source_overlap": {
                "sources": sorted(edition_sources),
                "compared_sections": len(edition_docs),
                "exact_duplicates": cross_exact,
                "near_duplicates": len(cross_near),
                "tokens_saved": edition_saved,
                "retained_source_priority": "book_edition",
            },
            "category_coverage": _distribution(eligible)["categories"],
            "classification": {
                "fallback_units": sum(
                    1 for doc in eligible if doc.metadata.get("classification_fallback")
                ),
                "fallback_tokens": sum(
                    doc.token_count or 0
                    for doc in eligible
                    if doc.metadata.get("classification_fallback")
                ),
                "unresolved_units_rejected": sum(
                    count.get("unresolved_classification", 0)
                    for count in rejection_reasons.values()
                ),
            },
            "score_distributions": {
                "relevance": _score_histogram(
                    [assessment.relevance_score for assessment in assessments],
                    (0.20, 0.28, 0.40, 0.60),
                ),
                "quality": _score_histogram(
                    [assessment.quality_score for assessment in assessments],
                    (0.25, 0.33, 0.45, 0.70),
                ),
                "joint_borderline": {
                    "both_in_rescue_window": sum(
                        assessment.relevance_score >= self.config.borderline_relevance_score
                        and assessment.quality_score >= self.config.borderline_quality_score
                        for assessment in assessments
                    ),
                    "rescued": sum(
                        assessment.decision == "rescued_borderline" for assessment in assessments
                    ),
                },
            },
            "rejection_samples": _rejection_samples(
                assessments, self.config.rejection_sample_size, self.config.seed
            ),
            **semantic,
        }

    def write_capacity(self, output_dir: str | Path) -> dict[str, Any]:
        result = self.capacity()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "capacity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        for name in (
            "concept_coverage",
            "category_coverage",
            "source_concept_matrix",
            "candidate_concepts",
        ):
            (out / f"{name}.json").write_text(
                json.dumps(result.get(name, {}), indent=2), encoding="utf-8"
            )
        (out / "source_diagnostics.json").write_text(
            json.dumps(result["source_capacity"], indent=2), encoding="utf-8"
        )
        write_dict_jsonl(result["rejection_samples"], out / "rejection_samples.jsonl")
        return result

    def _assert_freeze_preflight(self, target_tokens: int) -> dict[str, Any]:
        capacity = self.capacity()
        lower_bound = int(target_tokens * (1 - self.config.token_tolerance))
        failures: list[str] = []
        if capacity["theoretical_source_cap_capacity"] < target_tokens:
            failures.append("Configured source token caps cannot reach the target.")
        if capacity["expected_maximum_achievable_tokens"] < lower_bound:
            failures.append("Eligible source capacity is below the lower token tolerance bound.")
        for category, details in capacity["category_capacity"].items():
            min_required = int(
                target_tokens * max(0.0, details["target_share"] - self.config.category_tolerance)
            )
            if details["available_eligible_tokens"] < min_required:
                failures.append(f"Category '{category}' cannot satisfy its tolerance.")
        if failures:
            raise ValueError("Freeze preflight failed: " + " ".join(failures))
        return capacity

    def build(self, target_tokens: int, output_dir: str | Path, *, frozen: bool) -> dict[str, Any]:
        if target_tokens < 1:
            raise ValueError("target_tokens must be positive")
        freeze_preflight = self._assert_freeze_preflight(target_tokens) if frozen else None
        ledger: list[dict[str, Any]] = []
        raw_docs: list[CorpusDocument] = []
        policies = {source.id: source for source in self.config.source_configs}
        for source in self.config.source_configs:
            if not source.enabled:
                ledger.append(
                    {"decision": "source_disabled", "source_id": source.id, "reason": source.notes}
                )
                continue
            docs = get_adapter(source).ingest()
            if not docs:
                ledger.append(
                    {
                        "decision": "source_no_documents",
                        "source_id": source.id,
                        "reason": "No eligible files or license verification failed.",
                    }
                )
            raw_docs.extend(docs)

        cleaner = TextCleaner()
        boilerplate = BoilerplateCleaner()
        candidates: list[CorpusDocument] = []
        for raw in raw_docs:
            cleaned = _normalize_structural_headings(
                boilerplate.clean_document(cleaner.clean_document(raw))
            )
            source = policies[raw.source_id]
            cleaned = _strip_policy_sections(cleaned, source.strip_section_patterns)
            candidates.extend(
                annotate_document(_assign_section_category(section, source.section_category_rules))
                for section in _sectionize(cleaned, self.counter, self.config.max_section_tokens)
            )

        accepted, assessments = self._evaluate_candidates(candidates, policies)
        score_map: dict[str, float] = {}
        for assessment in assessments:
            enriched = assessment.document
            record = {
                "document_id": enriched.id,
                "source_id": enriched.source_id,
                "category": enriched.category,
                "relative_path": enriched.relative_path,
                "token_count": enriched.token_count,
                "quality_score": assessment.quality_score,
                "architecture_relevance_score": assessment.relevance_score,
                "code_ratio": enriched.code_ratio,
                "link_ratio": assessment.link_ratio,
            }
            if assessment.decision in {"normal_pass", "rescued_borderline"}:
                score_map[enriched.id] = (
                    assessment.quality_score
                    + assessment.relevance_score
                    + policies[enriched.source_id].source_priority / 100
                )
            ledger.append(
                {
                    **record,
                    "decision": assessment.decision,
                    "reason": assessment.reason,
                }
            )

        # Same-document grouping happens only after individual sections passed
        # quality gates.  It never crosses a document/source boundary.
        accepted = group_adjacent_sections(accepted, self.counter, self.config.max_section_tokens)
        accepted = [
            doc.model_copy(
                update={
                    "token_count": self.counter.count(doc.text),
                    "content_sha256": _content_hash(doc.text),
                }
            )
            for doc in accepted
        ]
        score_map = {
            doc.id: (doc.quality_score or 0.0)
            + (doc.architecture_relevance_score or 0.0)
            + policies[doc.source_id].source_priority / 100
            for doc in accepted
        }

        # Stable best-record exact deduplication: better scored source wins, then ID.
        ordered = sorted(accepted, key=lambda doc: (-score_map[doc.id], doc.id))
        exact = ExactDeduplicator().deduplicate(ordered)
        retained_exact = {doc.id for doc in exact.deduplicated_documents}
        exact_canonical_by_hash = {doc.content_sha256: doc for doc in exact.deduplicated_documents}
        for doc in ordered:
            if doc.id not in retained_exact:
                canonical = exact_canonical_by_hash.get(doc.content_sha256)
                ledger.append(
                    {
                        "document_id": doc.id,
                        "source_id": doc.source_id,
                        "relative_path": doc.relative_path,
                        "decision": "exact_duplicate",
                        "canonical_document_id": canonical.id if canonical else None,
                        "canonical_source_id": canonical.source_id if canonical else None,
                        "reason": "Identical normalized text; higher-ranked canonical record retained.",
                    }
                )

        near = MinHashLSHDeduplicator(similarity_threshold=0.85)
        near_result = near.deduplicate(exact.deduplicated_documents, score_map)
        exact_by_id = {doc.id: doc for doc in exact.deduplicated_documents}
        for cluster in near_result.clusters:
            for removed in cluster.removed_document_ids:
                removed_doc = exact_by_id[removed]
                canonical_doc = exact_by_id[cluster.canonical_document_id]
                ledger.append(
                    {
                        "document_id": removed,
                        "source_id": removed_doc.source_id,
                        "relative_path": removed_doc.relative_path,
                        "decision": "near_duplicate",
                        "cluster_id": cluster.cluster_id,
                        "canonical_document_id": cluster.canonical_document_id,
                        "canonical_source_id": canonical_doc.source_id,
                        "canonical_relative_path": canonical_doc.relative_path,
                        "similarity": cluster.estimated_similarity,
                        "reason": "Near-duplicate; retained canonical has the highest deterministic quality/relevance/priority rank.",
                    }
                )

        selected, balanced_out = self._select(
            near_result.canonical_documents,
            target_tokens,
            policies,
            allow_backfill=self.config.allow_preview_backfill
            if not frozen
            else self.config.allow_freeze_backfill,
        )
        if frozen:
            self._assert_freeze_selection(selected, target_tokens)
        for doc in selected:
            ledger.append(
                {
                    "document_id": doc.id,
                    "source_id": doc.source_id,
                    "relative_path": doc.relative_path,
                    "decision": "selected",
                    "reason": str(doc.metadata.get("balance_selection", "selected")),
                    "token_count": doc.token_count,
                    "primary_category": doc.primary_category or doc.category,
                    "related_concepts": doc.related_concepts,
                }
            )
        for doc, reason in balanced_out:
            ledger.append(
                {
                    "document_id": doc.id,
                    "source_id": doc.source_id,
                    "decision": "balanced_out",
                    "reason": reason,
                    "token_count": doc.token_count,
                }
            )
        splitter = GroupCorpusSplitter(*self.config.split_ratios, seed=self.config.seed)
        split = splitter.split(selected)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(split.train_documents, out / "train.jsonl")
        write_jsonl(split.validation_documents, out / "validation.jsonl")
        write_jsonl(split.heldout_documents, out / "heldout.jsonl")
        write_dict_jsonl(ledger, out / "audit.jsonl")
        semantic = coverage_report(
            selected,
            min_tokens=self.config.concept_min_tokens,
            min_sources=self.config.concept_min_sources,
            min_documents=self.config.concept_min_documents,
            max_dominant_source_share=self.config.max_concept_dominant_source_share,
        )
        for name, value in semantic.items():
            (out / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
        (out / "category_coverage.json").write_text(
            json.dumps(category_coverage_report(selected), indent=2), encoding="utf-8"
        )
        source_diagnostics = {
            source_id: {
                "training_units": len(items),
                "documents": len({item.relative_path or item.id for item in items}),
                "tokens": sum(item.token_count or 0 for item in items),
            }
            for source_id, items in sorted(
                (
                    (source_id, [doc for doc in selected if doc.source_id == source_id])
                    for source_id in sorted({doc.source_id for doc in selected})
                ),
            )
        }
        (out / "source_diagnostics.json").write_text(
            json.dumps(source_diagnostics, indent=2), encoding="utf-8"
        )
        license_report = self.license_audit()
        (out / "license_audit.json").write_text(
            json.dumps(license_report, indent=2), encoding="utf-8"
        )
        release_docs = [doc for doc in selected if bool(doc.metadata.get("release_eligible"))]
        manifest = self._manifest(target_tokens, selected, split, ledger, near_result, frozen)
        split_documents = {
            "train": split.train_documents,
            "validation": split.validation_documents,
            "heldout": split.heldout_documents,
        }
        corpus_fingerprint = compute_corpus_fingerprint(selected, manifest["config_hash"])
        try:
            build_git_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip() or "unknown"
        except OSError:
            build_git_sha = "unknown"
        manifest.update(
            {
                "artifact_type": "freeze" if frozen else "preview",
                "semantic_schema_version": 3,
                "freeze_preflight_passed": bool(freeze_preflight) if frozen else False,
                "actual_selected_token_count": sum(doc.token_count or 0 for doc in selected),
                "build_git_sha": build_git_sha,
                "deterministic_seed": self.config.seed,
                "split_ratios": {
                    "train": self.config.split_ratios[0],
                    "validation": self.config.split_ratios[1],
                    "heldout": self.config.split_ratios[2],
                },
                "corpus_fingerprint": corpus_fingerprint,
                "split_fingerprints": {
                    name: compute_corpus_fingerprint(docs, manifest["config_hash"])
                    for name, docs in split_documents.items()
                },
                "selected_source_ids": sorted({doc.source_id for doc in selected}),
                "concept_coverage_summary": {
                    "canonical_concepts": len(semantic["concept_coverage"]),
                    "warning_concepts": sorted(
                        concept
                        for concept, report in semantic["concept_coverage"].items()
                        if report["status"] != "healthy"
                    ),
                },
                "classification": {
                    "fallback_units": sum(
                        1 for doc in selected if doc.metadata.get("classification_fallback")
                    ),
                    "fallback_tokens": sum(
                        doc.token_count or 0
                        for doc in selected
                        if doc.metadata.get("classification_fallback")
                    ),
                    "unresolved_units_rejected": sum(
                        1
                        for record in ledger
                        if "unresolved classification" in str(record.get("reason", ""))
                    ),
                },
                "release_eligibility": {
                    "release_eligible_units": len(release_docs),
                    "release_ineligible_units": len(selected) - len(release_docs),
                    "release_eligible_tokens": sum(doc.token_count or 0 for doc in release_docs),
                },
                "artifact_hashes": {
                    name: _sha256_file(out / name)
                    for name in (
                        "audit.jsonl",
                        "concept_coverage.json",
                        "category_coverage.json",
                        "source_diagnostics.json",
                        "license_audit.json",
                    )
                },
            }
        )
        (out / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        experimental_manifest = {
            **manifest,
            "manifest_type": "experimental",
            "release_eligible": False,
        }
        release_manifest = {
            "manifest_type": "release_eligible",
            "corpus_version": self.config.corpus_version,
            "record_count": len(release_docs),
            "token_count": sum(doc.token_count or 0 for doc in release_docs),
            "source_distribution": _distribution(release_docs)["sources"],
            "category_distribution": _distribution(release_docs)["categories"],
            "excluded_experimental_records": len(selected) - len(release_docs),
        }
        (out / "experimental_manifest.json").write_text(
            json.dumps(experimental_manifest, indent=2), encoding="utf-8"
        )
        (out / "release_eligible_manifest.json").write_text(
            json.dumps(release_manifest, indent=2), encoding="utf-8"
        )
        return manifest

    def _select(
        self,
        docs: list[CorpusDocument],
        target_tokens: int,
        policies: dict[str, SourceConfig],
        *,
        allow_backfill: bool,
    ) -> tuple[list[CorpusDocument], list[tuple[CorpusDocument, str]]]:
        by_category: dict[str, list[CorpusDocument]] = defaultdict(list)
        concept_source_count = {
            concept: len({doc.source_id for doc in docs if concept in doc.related_concepts})
            for concept in {concept for doc in docs for concept in doc.related_concepts}
        }
        concept_source_tokens: dict[str, Counter[str]] = defaultdict(Counter)
        for doc in docs:
            for concept in doc.related_concepts:
                concept_source_tokens[concept][doc.source_id] += doc.token_count or 0

        def concept_order_key(doc: CorpusDocument) -> tuple[float, float, float, str]:
            """Prefer scarce concepts and less dominant sources deterministically."""
            quality_key = -((doc.quality_score or 0) + (doc.architecture_relevance_score or 0))
            if not self.config.concept_aware_selection or not doc.related_concepts:
                return (10_000.0, 1.0, quality_key, doc.id)
            source_counts = [concept_source_count[concept] for concept in doc.related_concepts]
            source_shares = [
                concept_source_tokens[concept][doc.source_id]
                / max(1, sum(concept_source_tokens[concept].values()))
                for concept in doc.related_concepts
            ]
            return (float(min(source_counts)), max(source_shares), quality_key, doc.id)

        for doc in docs:
            by_category[doc.primary_category or doc.category].append(doc)
        for documents in by_category.values():
            documents.sort(key=concept_order_key)
        category_limits = {
            category: int(target_tokens * target)
            for category, target in self.config.category_targets.items()
        }
        source_limits = {
            source_id: min(
                int(target_tokens * self.config.max_source_share),
                policy.source_token_cap if policy.source_token_cap else target_tokens,
            )
            for source_id, policy in policies.items()
        }
        selected: list[CorpusDocument] = []
        rejected: list[tuple[CorpusDocument, str]] = []
        totals: Counter[str] = Counter()
        source_totals: Counter[str] = Counter()
        positions: Counter[str] = Counter()
        while sum(doc.token_count or 0 for doc in selected) < target_tokens:
            available = [
                category
                for category in self.config.category_targets
                if positions[category] < len(by_category[category])
            ]
            if not available:
                break
            category = max(available, key=lambda key: (category_limits[key] - totals[key], key))
            doc = by_category[category][positions[category]]
            positions[category] += 1
            tokens = doc.token_count or 0
            if source_totals[doc.source_id] + tokens > source_limits[doc.source_id]:
                rejected.append((doc, "Source token cap reached."))
                continue
            if (
                totals[category] + tokens
                > category_limits[category] + self.config.max_section_tokens
            ):
                rejected.append((doc, "Category target reached."))
                continue
            if sum(item.token_count or 0 for item in selected) + tokens > target_tokens + int(
                target_tokens * self.config.token_tolerance
            ):
                rejected.append((doc, "Target token tolerance reached."))
                continue
            selection_reason = "category/source-balanced selection"
            if self.config.concept_aware_selection and doc.related_concepts:
                scarce = min(doc.related_concepts, key=lambda concept: (concept_source_count[concept], concept))
                selection_reason = (
                    "concept-aware selection: prioritised "
                    f"{scarce} ({concept_source_count[scarce]} eligible sources)"
                )
            selected.append(doc.model_copy(update={"metadata": {**doc.metadata, "balance_selection": selection_reason}}))
            totals[category] += tokens
            source_totals[doc.source_id] += tokens
        # Preview may use remaining capacity for diagnostics; final freeze does not.
        # use the remaining capacity of other categories rather than fabricate
        # data or silently fail to produce a useful preview.  Source caps remain
        # strict and the manifest exposes the resulting category shortfall.
        selected_ids = {doc.id for doc in selected}
        for doc in (
            sorted(
                docs,
                key=lambda item: (
                    -((item.quality_score or 0.0) + (item.architecture_relevance_score or 0.0)),
                    item.id,
                ),
            )
            if allow_backfill
            else []
        ):
            if (
                sum(item.token_count or 0 for item in selected) >= target_tokens
                or doc.id in selected_ids
            ):
                continue
            tokens = doc.token_count or 0
            if source_totals[doc.source_id] + tokens > source_limits[doc.source_id]:
                continue
            if sum(item.token_count or 0 for item in selected) + tokens > target_tokens + int(
                target_tokens * self.config.token_tolerance
            ):
                continue
            selected.append(
                doc.model_copy(
                    update={
                        "metadata": {
                            **doc.metadata,
                            "balance_selection": "preview backfill after category selection",
                        }
                    }
                )
            )
            selected_ids.add(doc.id)
            totals[doc.primary_category or doc.category] += tokens
            source_totals[doc.source_id] += tokens
        return selected, [(doc, reason) for doc, reason in rejected if doc.id not in selected_ids]

    def _assert_freeze_selection(self, docs: list[CorpusDocument], target_tokens: int) -> None:
        total = sum(doc.token_count or 0 for doc in docs)
        lower = int(target_tokens * (1 - self.config.token_tolerance))
        upper = int(target_tokens * (1 + self.config.token_tolerance))
        if not lower <= total <= upper:
            raise ValueError("Freeze selection is outside the configured token tolerance.")
        distributions = _distribution(docs)
        if any(
            float(item["share"]) > self.config.max_source_share + 0.0001
            for item in distributions["sources"].values()
        ):
            raise ValueError("Freeze selection exceeds the maximum source share.")
        for category, target in self.config.category_targets.items():
            actual = float(distributions["categories"].get(category, {}).get("share", 0.0))
            if abs(actual - target) > self.config.category_tolerance:
                raise ValueError(f"Freeze selection violates category tolerance for '{category}'.")
        hashes = [doc.content_sha256 for doc in docs]
        if len({doc.id for doc in docs}) != len(docs) or len(set(hashes)) != len(hashes):
            raise ValueError("Freeze selection contains duplicate IDs or exact content hashes.")

    def _manifest(
        self,
        target: int,
        docs: list[CorpusDocument],
        split: Any,
        ledger: list[dict[str, Any]],
        near: Any,
        frozen: bool,
    ) -> dict[str, Any]:
        total = sum(doc.token_count or 0 for doc in docs)
        decisions = Counter(str(item["decision"]) for item in ledger)
        corpus_hash = hashlib.sha256(
            "".join(sorted(doc.content_sha256 or "" for doc in docs)).encode()
        ).hexdigest()
        return {
            "corpus_version": self.config.corpus_version,
            "frozen": frozen,
            "config_hash": hashlib.sha256(self.config.config_path.read_bytes()).hexdigest(),
            "tokenizer": {
                "identifier": self.config.tokenizer_identifier,
                "revision": self.config.tokenizer_revision,
            },
            "target_tokens": target,
            "final_token_count": total,
            "corpus_hash": corpus_hash,
            "source_distribution": _distribution(docs)["sources"],
            "category_distribution": _distribution(docs)["categories"],
            "rejected_counts": dict(sorted(decisions.items())),
            "dedup": {
                "near_duplicate_clusters": len(near.clusters),
                "near_duplicate_documents": len(near.removed_documents),
            },
            "splits": {
                "train": {
                    "documents": len(split.train_documents),
                    "tokens": sum(doc.token_count or 0 for doc in split.train_documents),
                },
                "validation": {
                    "documents": len(split.validation_documents),
                    "tokens": sum(doc.token_count or 0 for doc in split.validation_documents),
                },
                "heldout": {
                    "documents": len(split.heldout_documents),
                    "tokens": sum(doc.token_count or 0 for doc in split.heldout_documents),
                },
            },
        }


def load_audit(path: str | Path) -> list[dict[str, Any]]:
    """Read a preview/freeze audit without treating the benchmark as corpus input."""
    audit_path = Path(path) / "audit.jsonl"
    return [
        json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line
    ]
