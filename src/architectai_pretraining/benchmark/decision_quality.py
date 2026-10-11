"""Evidence-grounded decision-status validation for the V2.1 prompt profile.

This module deliberately performs only deterministic grounding checks.  Whether an
unknown is truly material, whether two constraints conflict, and whether a
recommendation is architecturally sound remain semantic judgments.  Those cases
are surfaced for review instead of being silently treated as a pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, cast

from architectai_pretraining.benchmark.models import BenchmarkSample, RawOutput

DecisionStatus = Literal["RECOMMENDED", "CONDITIONAL", "BLOCKED"]
StaticCheckStatus = Literal["supported", "invalid", "unavailable"]

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
_NUMBER = re.compile(r"(?<![A-Za-z])(?:[$€£]\s*)?\d[\d,]*(?:\.\d+)?(?:\s*[kKmMgG])?(?:\s*%)?")

# This is intentionally a small, auditable vocabulary.  It catches names that
# commonly turn a capability-level decision into an unsupported product/topology
# commitment; it is not a claim to recognize every technology.
_NAMED_DECISION_TERMS = (
    "kubernetes",
    "kafka",
    "rabbitmq",
    "postgresql",
    "postgres",
    "mysql",
    "mongodb",
    "redis",
    "aws",
    "azure",
    "gcp",
    "google cloud",
    "amazon web services",
    "terraform",
    "istio",
    "raft",
    "paxos",
    "microservices",
    "modular monolith",
    "monolith",
    "event-driven",
    "event driven",
    "serverless",
)


@dataclass(frozen=True)
class EvidenceFact:
    """A model-declared fact and its mandatory verbatim scenario evidence."""

    claim: str
    scenario_evidence: str


@dataclass(frozen=True)
class UnknownDriver:
    """A model-declared unknown and why it is or is not material."""

    driver: str
    rationale: str


@dataclass(frozen=True)
class DeclaredConflict:
    """A model-declared conflict with evidence for both stated constraints."""

    description: str
    left_evidence: str
    right_evidence: str


@dataclass(frozen=True)
class ProvisionalDirection:
    """Whether a capability-level temporary direction is safe to give."""

    possible: bool
    direction: str


@dataclass(frozen=True)
class EvidenceCheck:
    """The structured evidence-check produced before the decision."""

    explicit_facts: list[EvidenceFact]
    material_unknowns: list[UnknownDriver]
    nonmaterial_unknowns: list[UnknownDriver]
    conflicts: list[DeclaredConflict]
    provisional_direction: ProvisionalDirection


@dataclass(frozen=True)
class DecisionSubmission:
    """One V2.1 decision response parsed from model output."""

    evidence_check: EvidenceCheck
    status: DecisionStatus
    recommendation: str


@dataclass(frozen=True)
class DecisionQualityRecord:
    """Deterministic grounding results, distinct from the legacy proxy score."""

    sample_id: str
    declared_status: str | None
    static_check_status: StaticCheckStatus
    requires_human_review: bool
    grounded_fact_count: int
    declared_fact_count: int
    grounded_number_count: int
    declared_number_count: int
    unsupported_named_terms: list[str]
    issues: list[str]
    semantic_review_flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "declared_status": self.declared_status,
            "static_check_status": self.static_check_status,
            "requires_human_review": self.requires_human_review,
            "grounded_fact_count": self.grounded_fact_count,
            "declared_fact_count": self.declared_fact_count,
            "grounded_number_count": self.grounded_number_count,
            "declared_number_count": self.declared_number_count,
            "unsupported_named_terms": self.unsupported_named_terms,
            "issues": self.issues,
            "semantic_review_flags": self.semantic_review_flags,
        }


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _source_text(sample: BenchmarkSample) -> str:
    return "\n".join([sample.scenario, *sample.facts])


def _as_mapping(value: object, path: str, errors: list[str]) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    errors.append(f"{path} must be an object.")
    return {}


def _as_string(value: object, path: str, errors: list[str]) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    errors.append(f"{path} must be a non-empty string.")
    return ""


def _as_list(value: object, path: str, errors: list[str]) -> list[object]:
    if isinstance(value, list):
        return value
    errors.append(f"{path} must be an array.")
    return []


def _extract_json_object(response_text: str) -> tuple[dict[str, object] | None, list[str]]:
    candidate = response_text.strip()
    fenced = _JSON_FENCE.fullmatch(candidate)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None, ["Response is not a single valid JSON object."]
    if not isinstance(value, dict):
        return None, ["Response JSON root must be an object."]
    return value, []


def parse_decision_submission(response_text: str) -> tuple[DecisionSubmission | None, list[str]]:
    """Parse the V2.1 schema without silently repairing malformed model output."""
    root, errors = _extract_json_object(response_text)
    if root is None:
        return None, errors

    evidence_root = _as_mapping(root.get("evidence_check"), "evidence_check", errors)
    decision_root = _as_mapping(root.get("decision"), "decision", errors)

    facts: list[EvidenceFact] = []
    for index, item in enumerate(_as_list(evidence_root.get("explicit_facts"), "explicit_facts", errors)):
        entry = _as_mapping(item, f"explicit_facts[{index}]", errors)
        facts.append(
            EvidenceFact(
                _as_string(entry.get("claim"), f"explicit_facts[{index}].claim", errors),
                _as_string(
                    entry.get("scenario_evidence"),
                    f"explicit_facts[{index}].scenario_evidence",
                    errors,
                ),
            )
        )

    def parse_unknowns(key: str, rationale_key: str) -> list[UnknownDriver]:
        unknowns: list[UnknownDriver] = []
        for index, item in enumerate(_as_list(evidence_root.get(key), key, errors)):
            entry = _as_mapping(item, f"{key}[{index}]", errors)
            unknowns.append(
                UnknownDriver(
                    _as_string(entry.get("driver"), f"{key}[{index}].driver", errors),
                    _as_string(entry.get(rationale_key), f"{key}[{index}].{rationale_key}", errors),
                )
            )
        return unknowns

    conflicts: list[DeclaredConflict] = []
    for index, item in enumerate(_as_list(evidence_root.get("conflicts"), "conflicts", errors)):
        entry = _as_mapping(item, f"conflicts[{index}]", errors)
        conflicts.append(
            DeclaredConflict(
                _as_string(entry.get("description"), f"conflicts[{index}].description", errors),
                _as_string(entry.get("left_evidence"), f"conflicts[{index}].left_evidence", errors),
                _as_string(entry.get("right_evidence"), f"conflicts[{index}].right_evidence", errors),
            )
        )

    provisional_root = _as_mapping(
        evidence_root.get("provisional_direction"), "provisional_direction", errors
    )
    possible = provisional_root.get("possible")
    if not isinstance(possible, bool):
        errors.append("provisional_direction.possible must be a boolean.")
        possible = False
    provisional = ProvisionalDirection(
        possible=possible,
        direction=_as_string(provisional_root.get("direction"), "provisional_direction.direction", errors),
    )

    status_value = _as_string(decision_root.get("status"), "decision.status", errors)
    if status_value not in {"RECOMMENDED", "CONDITIONAL", "BLOCKED"}:
        errors.append("decision.status must be RECOMMENDED, CONDITIONAL, or BLOCKED.")
    recommendation = _as_string(decision_root.get("recommendation"), "decision.recommendation", errors)

    material_unknowns = parse_unknowns("material_unknowns", "why_material")
    nonmaterial_unknowns = parse_unknowns("nonmaterial_unknowns", "why_not_material")
    if errors:
        return None, errors
    status = cast(DecisionStatus, status_value)
    return (
        DecisionSubmission(
            evidence_check=EvidenceCheck(
                explicit_facts=facts,
                material_unknowns=material_unknowns,
                nonmaterial_unknowns=nonmaterial_unknowns,
                conflicts=conflicts,
                provisional_direction=provisional,
            ),
            status=status,
            recommendation=recommendation,
        ),
        [],
    )


def _numbers(text: str) -> list[str]:
    return [_normalized(match.group(0)).replace(" ", "") for match in _NUMBER.finditer(text)]


def _all_submission_text(submission: DecisionSubmission) -> str:
    parts = [submission.recommendation]
    for fact in submission.evidence_check.explicit_facts:
        parts.extend([fact.claim, fact.scenario_evidence])
    for unknown in (
        *submission.evidence_check.material_unknowns,
        *submission.evidence_check.nonmaterial_unknowns,
    ):
        parts.extend([unknown.driver, unknown.rationale])
    for conflict in submission.evidence_check.conflicts:
        parts.extend([conflict.description, conflict.left_evidence, conflict.right_evidence])
    parts.append(submission.evidence_check.provisional_direction.direction)
    return "\n".join(parts)


def _unsupported_named_terms(text: str, source_text: str) -> list[str]:
    source = _normalized(source_text)
    response = _normalized(text)
    findings: list[str] = []
    for term in _NAMED_DECISION_TERMS:
        pattern = re.compile(rf"\b{re.escape(term)}\b")
        if pattern.search(response) and not pattern.search(source):
            findings.append(term)
    return findings


def validate_decision_submission(sample: BenchmarkSample, response_text: str) -> DecisionQualityRecord:
    """Validate grounding and status prerequisites without claiming semantic proof.

    `supported` means only that static checks found no violation.  Every record
    continues to require human review because materiality, conflict resolution,
    and architectural sufficiency cannot be established by string rules.
    """
    submission, parse_errors = parse_decision_submission(response_text)
    if submission is None:
        return DecisionQualityRecord(
            sample_id=sample.id,
            declared_status=None,
            static_check_status="unavailable",
            requires_human_review=True,
            grounded_fact_count=0,
            declared_fact_count=0,
            grounded_number_count=0,
            declared_number_count=0,
            unsupported_named_terms=[],
            issues=parse_errors,
            semantic_review_flags=["structured_evidence_unavailable"],
        )

    source = _normalized(_source_text(sample))
    issues: list[str] = []
    grounded_facts = 0
    for index, fact in enumerate(submission.evidence_check.explicit_facts):
        if _normalized(fact.scenario_evidence) in source:
            grounded_facts += 1
        else:
            issues.append(f"explicit_facts[{index}] is not a verbatim quote from the scenario.")

    for index, conflict in enumerate(submission.evidence_check.conflicts):
        if _normalized(conflict.left_evidence) not in source:
            issues.append(f"conflicts[{index}].left_evidence is not grounded in the scenario.")
        if _normalized(conflict.right_evidence) not in source:
            issues.append(f"conflicts[{index}].right_evidence is not grounded in the scenario.")

    declared_numbers = _numbers(_all_submission_text(submission))
    source_numbers = set(_numbers(_source_text(sample)))
    grounded_numbers = sum(number in source_numbers for number in declared_numbers)
    unsupported_numbers = sorted({number for number in declared_numbers if number not in source_numbers})
    if unsupported_numbers:
        issues.append("Unsupported numeric values: " + ", ".join(unsupported_numbers) + ".")

    unsupported_terms = _unsupported_named_terms(_all_submission_text(submission), _source_text(sample))
    if unsupported_terms:
        issues.append("Unsupported named technology/topology terms: " + ", ".join(unsupported_terms) + ".")

    evidence = submission.evidence_check
    if submission.status == "RECOMMENDED":
        if len(evidence.explicit_facts) < 2:
            issues.append("RECOMMENDED requires at least two declared explicit facts.")
        if evidence.material_unknowns:
            issues.append("RECOMMENDED cannot include material unknowns.")
        if evidence.conflicts:
            issues.append("RECOMMENDED cannot include unresolved conflicts.")
    elif submission.status == "CONDITIONAL":
        if not evidence.material_unknowns:
            issues.append("CONDITIONAL requires at least one material unknown.")
        if not evidence.provisional_direction.possible:
            issues.append("CONDITIONAL requires a safe provisional direction.")
    else:  # BLOCKED
        if not evidence.conflicts:
            issues.append("BLOCKED requires a directly evidenced conflict.")
        if evidence.provisional_direction.possible:
            issues.append("BLOCKED is inconsistent with a declared safe provisional direction.")

    semantic_flags = ["materiality_and_conflict_semantics_not_proven_by_static_checks"]
    if submission.status == "RECOMMENDED":
        semantic_flags.append("recommendation_sufficiency_requires_human_review")
    if evidence.material_unknowns or evidence.nonmaterial_unknowns:
        semantic_flags.append("unknown_materiality_requires_human_review")
    if evidence.conflicts:
        semantic_flags.append("conflict_resolvability_requires_human_review")

    return DecisionQualityRecord(
        sample_id=sample.id,
        declared_status=submission.status,
        static_check_status="supported" if not issues else "invalid",
        requires_human_review=True,
        grounded_fact_count=grounded_facts,
        declared_fact_count=len(evidence.explicit_facts),
        grounded_number_count=grounded_numbers,
        declared_number_count=len(declared_numbers),
        unsupported_named_terms=unsupported_terms,
        issues=issues,
        semantic_review_flags=semantic_flags,
    )


def decision_quality_summary(records: list[DecisionQualityRecord]) -> dict[str, Any]:
    """Produce V2.1 decision-quality telemetry without using proxy-score inputs."""
    total = len(records)
    declared_facts = sum(record.declared_fact_count for record in records)
    declared_numbers = sum(record.declared_number_count for record in records)
    status_counts = dict.fromkeys(("RECOMMENDED", "CONDITIONAL", "BLOCKED"), 0)
    for record in records:
        if record.declared_status in status_counts:
            status_counts[record.declared_status] += 1
    return {
        "evaluator_type": "decision_quality_static_v2_1",
        "total": total,
        "parseable_rate": round(
            sum(record.static_check_status != "unavailable" for record in records) / max(1, total), 4
        ),
        "static_supported_rate": round(
            sum(record.static_check_status == "supported" for record in records) / max(1, total), 4
        ),
        "review_required_rate": round(
            sum(record.requires_human_review for record in records) / max(1, total), 4
        ),
        "grounded_fact_rate": round(
            sum(record.grounded_fact_count for record in records) / max(1, declared_facts), 4
        ),
        "grounded_number_rate": round(
            sum(record.grounded_number_count for record in records) / max(1, declared_numbers), 4
        ),
        "unsupported_named_term_count": sum(len(record.unsupported_named_terms) for record in records),
        "status_counts": status_counts,
    }


def evaluate_decision_quality(
    samples: list[BenchmarkSample], raw_outputs: list[RawOutput]
) -> tuple[list[DecisionQualityRecord], dict[str, Any]]:
    """Evaluate saved V2.1 outputs without invoking a model or proxy evaluator."""
    raw_by_id = {output.sample_id: output for output in raw_outputs}
    records = [
        validate_decision_submission(sample, raw_by_id[sample.id].raw_response)
        for sample in samples
        if sample.id in raw_by_id
    ]
    return records, decision_quality_summary(records)
