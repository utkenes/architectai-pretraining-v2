"""Deterministic heuristic rubric evaluator for benchmark responses.

IMPORTANT: Deterministic rubric scores are explicitly labeled as PROXY/HEURISTIC metrics
(evaluator_type="deterministic_proxy_v1"). Keyword and rule scoring is a structured proxy
and must NOT be claimed as equivalent to expert human architecture evaluation.
"""

import re
from pathlib import Path
from typing import Any

from architectai_pretraining.benchmark.models import BenchmarkSample, EvaluationResult, RawOutput

# Absolute / unsupported claim patterns that penalize unsupported_claim_avoidance
UNSUPPORTED_CLAIM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"microservices?\s+(?:are\s+)?always\s+(?:required|needed|better)", re.IGNORECASE),
        "Absolute claim: microservices are always required",
    ),
    (
        re.compile(r"kubernetes?\s+automatically\s+provides\s+high\s+availability", re.IGNORECASE),
        "Absolute claim: Kubernetes automatically provides high availability",
    ),
    (
        re.compile(r"nosql\s+is\s+(?:always\s+)?faster\s+than\s+sql", re.IGNORECASE),
        "Absolute claim: NoSQL is faster than SQL",
    ),
    (
        re.compile(r"event-driven\s+architecture\s+guarantees\s+scalability", re.IGNORECASE),
        "Absolute claim: event-driven architecture guarantees scalability",
    ),
    (
        re.compile(r"always\s+(?:use|choose)\s+microservices", re.IGNORECASE),
        "Absolute claim: always use microservices",
    ),
    (
        re.compile(r"never\s+use\s+a?\s*monolith", re.IGNORECASE),
        "Absolute claim: never use a monolith",
    ),
]

# Concrete operational revisit signals
REVISIT_SIGNAL_KEYWORDS: list[str] = [
    "p95", "p99", "latency", "cpu", "memory", "i/o", "io", "connection pool",
    "queue lag", "incident", "failure rate", "team ownership", "deployment frequency",
    "throughput", "rps", "qps", "db load", "revisit", "threshold"
]

# Trade-off keywords
TRADEOFF_KEYWORDS: list[str] = [
    "tradeoff", "trade-off", "however", "on the other hand", "downside", "drawback",
    "benefit", "cost", "compromise", "versus", "vs"
]


class DeterministicRubricEvaluator:
    """Evaluates benchmark responses using rule-based heuristic proxies."""

    EVALUATOR_TYPE = "deterministic_proxy_v1"

    def evaluate(self, sample: BenchmarkSample, raw_output: RawOutput) -> EvaluationResult:
        text = raw_output.raw_response.lower()
        scores_dict: dict[str, float] = {}

        # 1. Driver identification proxy score
        facts_found = sum(1 for f in sample.facts if any(w in text for w in f.lower().split() if len(w) > 4))
        scores_dict["driver_identification"] = round(facts_found / max(1, len(sample.facts)), 2)

        # 2. NFR coverage proxy score
        nfr_terms = ["availability", "latency", "scalability", "consistency", "security", "cost", "reliability"]
        nfr_found = sum(1 for term in nfr_terms if term in text)
        scores_dict["nfr_coverage"] = round(min(1.0, nfr_found / 4.0), 2)

        # 3. Architecture choice quality proxy score
        arch_terms = [
            "monolith", "microservices", "modular monolith", "event-driven", "cqrs",
            "event sourcing", "hexagonal", "layered", "serverless", "crud"
        ]
        has_choice = any(term in text for term in arch_terms)
        scores_dict["architecture_choice_quality"] = 0.8 if has_choice else 0.2

        # 4. Tradeoff quality proxy score
        tradeoff_count = sum(1 for kw in TRADEOFF_KEYWORDS if kw in text)
        scores_dict["tradeoff_quality"] = round(min(1.0, tradeoff_count / 3.0), 2)

        # 5. Alternative analysis proxy score
        alt_words = ["alternative", "option", "instead of", "could also", "consider"]
        scores_dict["alternative_analysis"] = 0.8 if any(w in text for w in alt_words) else 0.2

        # 6. Risk identification proxy score
        risk_words = ["risk", "failure", "single point of failure", "spof", "bottleneck", "complexity"]
        scores_dict["risk_identification"] = 0.8 if any(w in text for w in risk_words) else 0.2

        # 7. Reliability reasoning proxy score
        rel_words = ["retry", "circuit breaker", "timeout", "bulkhead", "outbox", "dlq", "dead letter", "rate limit", "backoff"]
        rel_count = sum(1 for w in rel_words if w in text)
        scores_dict["reliability_reasoning"] = round(min(1.0, rel_count / 2.0), 2)

        # 8. Scalability reasoning proxy score
        scale_words = ["partition", "shard", "cache", "read replica", "index", "load balancer", "horizontal"]
        scale_count = sum(1 for w in scale_words if w in text)
        scores_dict["scalability_reasoning"] = round(min(1.0, scale_count / 2.0), 2)

        # 9. Operational complexity awareness proxy score
        ops_words = ["deployment", "operational", "monitoring", "observability", "team skill", "maintenance"]
        scores_dict["operational_complexity_awareness"] = 0.8 if any(w in text for w in ops_words) else 0.2

        # 10. Cost awareness proxy score
        cost_words = ["cost", "budget", "expensive", "cloud spend", "financial", "billing"]
        scores_dict["cost_awareness"] = 0.8 if any(w in text for w in cost_words) else 0.2

        # 11. Unsupported claim avoidance & detection
        unsupported_claims: list[str] = []
        for pattern, explanation in UNSUPPORTED_CLAIM_PATTERNS:
            if pattern.search(raw_output.raw_response):
                unsupported_claims.append(explanation)

        claim_avoidance_score = max(0.0, 1.0 - (len(unsupported_claims) * 0.5))
        scores_dict["unsupported_claim_avoidance"] = round(claim_avoidance_score, 2)
        unsupported_claim_rate = round(len(unsupported_claims) / len(UNSUPPORTED_CLAIM_PATTERNS), 4)

        # 12. Revisit conditions proxy score
        revisit_found = [kw for kw in REVISIT_SIGNAL_KEYWORDS if kw in text]
        scores_dict["revisit_conditions"] = round(min(1.0, len(revisit_found) / 3.0), 2)

        # 13. Clarification awareness proxy score
        clarification_words = ["clarify", "unknown", "missing information", "depends on", "unspecified", "budget is not stated"]
        ack_missing = any(w in text for w in clarification_words)
        scores_dict["clarification_awareness"] = 0.9 if ack_missing else (0.4 if sample.metadata.get("has_missing_info") else 0.7)

        # 14. Reasoning completeness proxy score
        word_count = len(text.split())
        length_score = min(1.0, word_count / 250.0)
        scores_dict["reasoning_completeness"] = round(length_score, 2)

        # Calculate overall proxy score (weighted average)
        total_score = sum(scores_dict.values())
        overall_proxy = round(total_score / len(scores_dict), 4)

        return EvaluationResult(
            sample_id=sample.id,
            category=sample.category,
            difficulty=sample.difficulty,
            scores=scores_dict,
            overall_proxy_score=overall_proxy,
            unsupported_claim_rate=unsupported_claim_rate,
            unsupported_claims=unsupported_claims,
            revisit_conditions_identified=revisit_found,
            missing_info_acknowledged=ack_missing,
            is_mock_eval=raw_output.is_mock,
            evaluator_type=self.EVALUATOR_TYPE,
        )


def export_human_review(
    samples: list[BenchmarkSample],
    raw_outputs: list[RawOutput],
    evaluations: list[EvaluationResult],
    output_path: str | Path,
) -> Path:
    """Generates human_review.jsonl for manual auditing of benchmark scores."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    samples_map = {s.id: s for s in samples}
    raw_map = {r.sample_id: r for r in raw_outputs}

    review_records: list[dict[str, Any]] = []
    for ev in evaluations:
        s = samples_map.get(ev.sample_id)
        raw = raw_map.get(ev.sample_id)
        if not s or not raw:
            continue

        review_records.append(
            {
                "sample_id": ev.sample_id,
                "category": ev.category,
                "difficulty": ev.difficulty,
                "scenario": s.scenario,
                "question": s.question,
                "facts": s.facts,
                "expected_considerations": s.expected_considerations,
                "base_model_response": raw.raw_response,
                "proxy_evaluator": ev.evaluator_type,
                "overall_proxy_score": ev.overall_proxy_score,
                "dimension_proxy_scores": ev.scores,
                "unsupported_claims": ev.unsupported_claims,
                "human_review_fields": {
                    "human_audited": False,
                    "auditor_notes": "",
                    "human_score_override": None,
                    "rubric_validity_confirmed": True,
                    "absence_of_answer_leakage": True,
                },
            }
        )

    from architectai_pretraining.io import write_dict_jsonl

    write_dict_jsonl(review_records, out_p)
    return out_p

# Benchmark scoring.py module update
