"""Unit tests for document and source quality scoring."""

from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.scoring import DocumentQualityScorer, calculate_source_quality_score


def test_document_quality_scorer_high_quality() -> None:
    scorer = DocumentQualityScorer(min_document_score=0.45)
    doc = CorpusDocument(
        id="test_arch_doc",
        title="Microservices Architecture Trade-offs",
        text="""# Microservices Architecture Trade-offs

This document details the architectural decisions and latency trade-offs between monolithic and microservice topologies.
Consistency vs availability requirements dictate partitioning and circuit breaker fault tolerance mechanisms.

## Trade-off Analysis
- Latency: Microservice IPC introduces network latency overhead.
- Availability: Decoupled components prevent cascading failures.
""",
        source_id="test_src",
        category="architecture_patterns",
        license_id="CC-BY-4.0",
        language="en",
    )

    score_res = scorer.score(doc)
    assert score_res.quality_score >= 0.50
    assert score_res.quality_bucket in ("high", "medium")
    assert any("trade-off" in r or "technical" in r for r in score_res.quality_reasons)


def test_document_quality_scorer_low_quality() -> None:
    scorer = DocumentQualityScorer(min_document_score=0.45)
    doc = CorpusDocument(
        id="test_low_doc",
        title="Random Text",
        text="$$$ %%% ### @@@ !!! +++ --- *** /// \\\\\\ foo bar baz 123 456 test random text without content.",
        source_id="test_src",
        category="architecture_patterns",
        license_id="CC-BY-4.0",
        language="en",
    )

    score_res = scorer.score(doc)
    assert score_res.quality_bucket == "low"


def test_calculate_source_quality_score() -> None:
    scorer = DocumentQualityScorer()
    doc = CorpusDocument(
        id="doc1",
        title="System Design",
        text="# System Design Architecture\nHigh availability distributed storage consensus.",
        source_id="src1",
        category="distributed_systems",
        license_id="MIT",
        language="en",
    )
    s1 = scorer.score(doc)
    src_score = calculate_source_quality_score("src1", [s1])
    assert src_score.score > 0.0
    assert src_score.source_id == "src1"


