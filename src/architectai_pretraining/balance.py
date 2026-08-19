"""Deterministic, quality-aware final-corpus concentration balancing.

Caps are evaluated against the retained corpus, never against the pre-filter total.
When document granularity makes a cap impossible, the result records an explicit
unsatisfied constraint instead of claiming that balancing succeeded.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.scoring import DocumentQualityScore


@dataclass
class ConcentrationMetrics:
    total_tokens: int
    top_1_source_share: float
    top_5_source_share: float
    top_category_share: float
    top_organization_share: float
    hhi_source_index: float
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BalanceConstraintStatus:
    dimension: str
    cap: float
    actual_share: float
    satisfied: bool
    reason: str | None = None


@dataclass
class BalancingResult:
    kept_documents: list[CorpusDocument]
    balanced_out_documents: list[CorpusDocument]
    concentration_before: ConcentrationMetrics
    concentration_after: ConcentrationMetrics
    removals_by_source: dict[str, int] = field(default_factory=dict)
    constraint_statuses: list[BalanceConstraintStatus] = field(default_factory=list)

    @property
    def constraints_satisfied(self) -> bool:
        return all(status.satisfied for status in self.constraint_statuses)


def _organization(doc: CorpusDocument) -> str:
    return str(doc.metadata.get("repository_owner", doc.source_id))


def _token_totals(
    docs: Sequence[CorpusDocument], doc_tokens: dict[str, int]
) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {"source": {}, "category": {}, "organization": {}}
    for doc in docs:
        tokens = doc_tokens.get(doc.id, 0)
        for dimension, value in (
            ("source", doc.source_id),
            ("category", doc.category),
            ("organization", _organization(doc)),
        ):
            bucket = totals[dimension]
            bucket[value] = bucket.get(value, 0) + tokens
    return totals


def calculate_concentration_metrics(
    docs: Sequence[CorpusDocument],
    doc_tokens: dict[str, int],
    category_targets: dict[str, dict[str, float]] | None = None,
) -> ConcentrationMetrics:
    total_tokens = sum(doc_tokens.get(doc.id, 0) for doc in docs)
    if total_tokens == 0:
        return ConcentrationMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, ["Corpus contains zero tokens."])
    totals = _token_totals(docs, doc_tokens)
    source_tokens, category_tokens, org_tokens = (
        totals["source"], totals["category"], totals["organization"]
    )
    sorted_sources = sorted(source_tokens.values(), reverse=True)
    top_source = sorted_sources[0] / total_tokens
    top_category = max(category_tokens.values()) / total_tokens
    top_org = max(org_tokens.values()) / total_tokens
    warnings: list[str] = []
    if top_source > 0.35:
        warnings.append(f"High source dominance: Top source holds {top_source:.1%} of tokens.")
    if top_category > 0.45:
        warnings.append(f"High category dominance: Top category holds {top_category:.1%} of tokens.")
    if category_targets:
        for category, target in category_targets.items():
            minimum = target.get("min_token_share", 0.0)
            actual = category_tokens.get(category, 0) / total_tokens
            if actual < minimum:
                warnings.append(
                    f"Underrepresented category '{category}': holds {actual:.1%} (min target: {minimum:.1%})."
                )
    return ConcentrationMetrics(
        total_tokens=total_tokens,
        top_1_source_share=round(top_source, 4),
        top_5_source_share=round(sum(sorted_sources[:5]) / total_tokens, 4),
        top_category_share=round(top_category, 4),
        top_organization_share=round(top_org, 4),
        hhi_source_index=round(sum((value / total_tokens) ** 2 for value in source_tokens.values()), 4),
        warnings=warnings,
    )


class CorpusBalancer:
    """Remove lowest-ranked documents while reducing final-corpus cap violations."""

    def __init__(
        self,
        max_source_token_share: float = 0.30,
        max_category_token_share: float = 0.40,
        max_organization_token_share: float = 0.40,
        category_targets: dict[str, dict[str, float]] | None = None,
        tolerance: float = 0.001,
    ) -> None:
        for value in (max_source_token_share, max_category_token_share, max_organization_token_share):
            if not 0.0 < value <= 1.0:
                raise ValueError("Balancing caps must be in (0, 1].")
        self.max_source_token_share = max_source_token_share
        self.max_category_token_share = max_category_token_share
        self.max_organization_token_share = max_organization_token_share
        self.category_targets = category_targets or {}
        self.tolerance = tolerance

    def _excess(self, docs: Sequence[CorpusDocument], doc_tokens: dict[str, int]) -> float:
        total = sum(doc_tokens.get(doc.id, 0) for doc in docs)
        if total == 0:
            return float("inf")
        return self._excess_from_totals(_token_totals(docs, doc_tokens), total)

    def _excess_from_totals(self, totals: dict[str, dict[str, int]], total: int) -> float:
        if total <= 0:
            return float("inf")
        caps = {
            "source": self.max_source_token_share,
            "category": self.max_category_token_share,
            "organization": self.max_organization_token_share,
        }
        return sum(
            max(0.0, (value / total) - caps[dimension])
            for dimension, groups in totals.items()
            for value in groups.values()
        )

    def _excess_after_removal(
        self, totals: dict[str, dict[str, int]], total: int, candidate: CorpusDocument, tokens: int
    ) -> float:
        """Score a removal without rebuilding a corpus list for every candidate."""
        if total <= tokens:
            return float("inf")
        adjusted = {dimension: groups.copy() for dimension, groups in totals.items()}
        for dimension, value in (
            ("source", candidate.source_id),
            ("category", candidate.category),
            ("organization", _organization(candidate)),
        ):
            adjusted[dimension][value] -= tokens
            if adjusted[dimension][value] == 0:
                del adjusted[dimension][value]
        return self._excess_from_totals(adjusted, total - tokens)

    def _statuses(
        self, docs: Sequence[CorpusDocument], doc_tokens: dict[str, int]
    ) -> list[BalanceConstraintStatus]:
        total = sum(doc_tokens.get(doc.id, 0) for doc in docs)
        if total == 0:
            return [
                BalanceConstraintStatus("source", self.max_source_token_share, 0.0, False, "No retained tokens."),
                BalanceConstraintStatus("category", self.max_category_token_share, 0.0, False, "No retained tokens."),
                BalanceConstraintStatus("organization", self.max_organization_token_share, 0.0, False, "No retained tokens."),
            ]
        totals = _token_totals(docs, doc_tokens)
        caps = {
            "source": self.max_source_token_share,
            "category": self.max_category_token_share,
            "organization": self.max_organization_token_share,
        }
        statuses: list[BalanceConstraintStatus] = []
        for dimension, cap in caps.items():
            actual = max(totals[dimension].values(), default=0) / total
            satisfied = actual <= cap + self.tolerance
            reason = None if satisfied else "Document-level granularity prevents further beneficial removal."
            statuses.append(BalanceConstraintStatus(dimension, cap, actual, satisfied, reason))
        return statuses

    def balance(
        self,
        docs: Sequence[CorpusDocument],
        doc_tokens: dict[str, int],
        doc_quality_scores: dict[str, DocumentQualityScore],
    ) -> BalancingResult:
        before = calculate_concentration_metrics(docs, doc_tokens, self.category_targets)
        kept = sorted(docs, key=lambda doc: doc.id)
        removed: list[CorpusDocument] = []
        removals_by_source: dict[str, int] = {}
        total = before.total_tokens
        totals = _token_totals(kept, doc_tokens)
        while kept:
            current_excess = self._excess_from_totals(totals, total)
            if current_excess <= self.tolerance:
                break
            best: CorpusDocument | None = None
            best_excess = current_excess
            for candidate in kept:
                candidate_tokens = doc_tokens.get(candidate.id, 0)
                candidate_excess = self._excess_after_removal(
                    totals, total, candidate, candidate_tokens
                )
                if candidate_excess + 1e-12 >= best_excess:
                    continue
                if best is None:
                    best, best_excess = candidate, candidate_excess
                    continue
                candidate_quality = doc_quality_scores.get(candidate.id, DocumentQualityScore(0.0, "low")).quality_score
                best_quality = doc_quality_scores.get(best.id, DocumentQualityScore(0.0, "low")).quality_score
                candidate_key = (candidate_quality, -doc_tokens.get(candidate.id, 0), candidate.id)
                best_key = (best_quality, -doc_tokens.get(best.id, 0), best.id)
                if candidate_key < best_key:
                    best, best_excess = candidate, candidate_excess
            if best is None:
                break
            kept.remove(best)
            removed.append(best)
            removed_tokens = doc_tokens.get(best.id, 0)
            total -= removed_tokens
            for dimension, value in (
                ("source", best.source_id),
                ("category", best.category),
                ("organization", _organization(best)),
            ):
                totals[dimension][value] -= removed_tokens
                if totals[dimension][value] == 0:
                    del totals[dimension][value]
            removals_by_source[best.source_id] = removals_by_source.get(best.source_id, 0) + 1
        after = calculate_concentration_metrics(kept, doc_tokens, self.category_targets)
        statuses = self._statuses(kept, doc_tokens)
        for status in statuses:
            if not status.satisfied:
                after.warnings.append(
                    f"UNSATISFIED {status.dimension} cap: {status.actual_share:.1%} > {status.cap:.1%}. {status.reason}"
                )
        return BalancingResult(kept, removed, before, after, removals_by_source, statuses)
