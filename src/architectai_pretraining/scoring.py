"""Deterministic source and document quality scoring models and heuristic engines."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.scoring_context import contextual_scoring_view

TECH_VOCAB = {
    "architecture",
    "architectural",
    "pattern",
    "topology",
    "distributed",
    "consensus",
    "replication",
    "partition",
    "sharding",
    "transaction",
    "consistency",
    "availability",
    "scalability",
    "throughput",
    "latency",
    "redundancy",
    "failover",
    "microservice",
    "monolith",
    "decoupling",
    "bounded",
    "aggregate",
    "domain",
    "cqrs",
    "outbox",
    "pubsub",
    "event",
    "messaging",
    "database",
    "cache",
    "storage",
    "resilience",
    "fault",
    "circuit",
    "protocol",
    "component",
    "interface",
    "api",
    "pipeline",
    "stream",
}

TRADEOFF_KEYWORDS = {
    "tradeoff",
    "trade-off",
    "alternative",
    "alternatives",
    "advantage",
    "disadvantage",
    "benefit",
    "drawback",
    "compromise",
    "overhead",
    "bottleneck",
    "rationale",
    "decision",
    "risk",
    "limitation",
    "consequence",
    "versus",
    "vs",
    "pro",
    "con",
}


@dataclass
class SourceQualityScore:
    source_id: str
    score: float
    quality_bucket: str  # "high", "medium", "low"
    reasons: list[str]
    metrics: dict[str, float]


@dataclass
class DocumentQualityScore:
    quality_score: float
    quality_bucket: str  # "high", "medium", "low"
    quality_reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class DocumentQualityScorer:
    """Deterministic document-level quality scoring heuristic engine."""

    def __init__(
        self,
        min_document_score: float = 0.45,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.min_document_score = min_document_score
        self.weights = weights or {
            "technical_vocab_density": 0.25,
            "tradeoff_keyword_density": 0.25,
            "natural_prose_density": 0.30,
            "heading_structure_density": 0.15,
            "code_to_prose_balance": 0.05,
        }

    def score(self, doc: CorpusDocument, token_count: int | None = None) -> DocumentQualityScore:
        text = doc.text
        scoring_text = contextual_scoring_view(doc)
        words = re.findall(r"\b[a-zA-Z]+\b", scoring_text.lower())
        word_count = len(words)
        if word_count == 0:
            return DocumentQualityScore(
                quality_score=0.0,
                quality_bucket="low",
                quality_reasons=["Empty or non-text document"],
                metrics={"word_count": 0.0},
            )

        # 1. Technical Vocabulary Density
        tech_matches = sum(1 for w in words if w in TECH_VOCAB)
        tech_density = min(1.0, (tech_matches / word_count) * 15.0)

        # 2. Architecture Trade-off Density
        tradeoff_matches = sum(1 for w in words if w in TRADEOFF_KEYWORDS)
        tradeoff_density = min(1.0, (tradeoff_matches / word_count) * 30.0)

        # 3. Natural Prose Density
        char_count = len(text)
        alpha_count = sum(1 for c in text if c.isalpha() or c.isspace())
        prose_density = alpha_count / max(1, char_count)

        # 4. Heading Structure Density
        # Markdown and AsciiDoc both carry meaningful document structure.
        heading_count = len(re.findall(r"^(?:#{1,6}\s+|={1,6}\s+)", scoring_text, re.MULTILINE))
        heading_density = min(1.0, heading_count / 3.0)

        # 5. Code vs Prose Balance
        code_blocks = re.findall(r"```[\s\S]*?```|\[source[^\]]*\]\s*----[\s\S]*?----", text)
        code_chars = sum(len(b) for b in code_blocks)
        code_ratio = code_chars / max(1, char_count)
        balance_score = 1.0 - min(1.0, code_ratio * 1.2)

        metrics = {
            "technical_vocab_density": round(tech_density, 4),
            "tradeoff_keyword_density": round(tradeoff_density, 4),
            "natural_prose_density": round(prose_density, 4),
            "heading_structure_density": round(heading_density, 4),
            "code_to_prose_balance": round(balance_score, 4),
            "code_ratio": round(code_ratio, 4),
            "word_count": float(word_count),
        }

        score = (
            tech_density * self.weights.get("technical_vocab_density", 0.25)
            + tradeoff_density * self.weights.get("tradeoff_keyword_density", 0.25)
            + prose_density * self.weights.get("natural_prose_density", 0.20)
            + heading_density * self.weights.get("heading_structure_density", 0.15)
            + balance_score * self.weights.get("code_to_prose_balance", 0.15)
        )
        score = round(min(1.0, max(0.0, score)), 4)

        reasons = []
        if tech_density >= 0.5:
            reasons.append("high technical vocabulary density")
        if tradeoff_density >= 0.3:
            reasons.append("contains architecture trade-off language")
        if prose_density >= 0.8:
            reasons.append("substantial technical prose")
        if heading_density >= 0.3:
            reasons.append("structured heading organization")
        if code_ratio > 0.6:
            reasons.append("high code block ratio")

        if score >= 0.70:
            bucket = "high"
        elif score >= self.min_document_score:
            bucket = "medium"
        else:
            bucket = "low"
            if not reasons:
                reasons.append("low technical prose density")

        return DocumentQualityScore(
            quality_score=score,
            quality_bucket=bucket,
            quality_reasons=reasons,
            metrics=metrics,
        )


def calculate_source_quality_score(
    source_id: str, doc_scores: Sequence[DocumentQualityScore]
) -> SourceQualityScore:
    if not doc_scores:
        return SourceQualityScore(
            source_id=source_id,
            score=0.0,
            quality_bucket="low",
            reasons=["No documents available for source"],
            metrics={"doc_count": 0.0},
        )

    avg_score = sum(ds.quality_score for ds in doc_scores) / len(doc_scores)
    high_ratio = sum(1 for ds in doc_scores if ds.quality_bucket == "high") / len(doc_scores)

    if avg_score >= 0.70:
        bucket = "high"
    elif avg_score >= 0.45:
        bucket = "medium"
    else:
        bucket = "low"

    reasons = [f"Average document score {avg_score:.2f}", f"High quality ratio {high_ratio:.1%}"]
    return SourceQualityScore(
        source_id=source_id,
        score=round(avg_score, 4),
        quality_bucket=bucket,
        reasons=reasons,
        metrics={
            "avg_document_score": round(avg_score, 4),
            "high_quality_ratio": round(high_ratio, 4),
            "doc_count": float(len(doc_scores)),
        },
    )
