"""Unit tests for training corpus contamination checking and ingestion exclusion rules."""

from pathlib import Path

import pytest

from architectai_pretraining.benchmark.contamination import (
    check_benchmark_against_corpus,
    is_benchmark_path_excluded,
)
from architectai_pretraining.benchmark.dataset import BenchmarkDataset
from architectai_pretraining.benchmark.models import BenchmarkSample
from architectai_pretraining.sources import matches_patterns


def test_is_benchmark_path_excluded() -> None:
    assert is_benchmark_path_excluded("data/benchmark/architectai_v1.jsonl")
    assert is_benchmark_path_excluded("data/benchmark/results/baseline/raw_outputs.jsonl")
    assert is_benchmark_path_excluded(Path("d:/architectai-pretraining/data/benchmark/architectai_v1.jsonl"))

    assert not is_benchmark_path_excluded("data/final/curated/train.jsonl")
    assert not is_benchmark_path_excluded("data/raw/manual/pattern.md")


def test_sources_matches_patterns_rejects_benchmark_files() -> None:
    bench_file = Path("data/benchmark/architectai_v1.jsonl")
    assert not matches_patterns(bench_file, Path("data"))


def test_check_benchmark_against_corpus(tmp_path: pytest.TempPathFactory) -> None:
    corpus_dir = tmp_path / "curated"  # type: ignore[operator]
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # 1. Non-contaminated corpus document
    clean_doc = {
        "id": "clean_doc_1",
        "text": "This document describes general database indexing strategies for PostgreSQL B-tree indices.",
    }

    # 2. Contaminated document sharing exact scenario text
    scen_text = "A 5-person engineering team at a Fintech startup is building an MVP payments reconciliation service."
    contam_doc = {
        "id": "contam_doc_1",
        "text": f"Overview: {scen_text} Additional text regarding payments.",
    }

    from architectai_pretraining.io import write_dict_jsonl

    write_dict_jsonl([clean_doc, contam_doc], corpus_dir / "train.jsonl")

    sample1 = BenchmarkSample(
        id="archbench_001",
        category="microservices_vs_monolith",
        difficulty="easy",
        scenario=scen_text,
        question="Should they build microservices?",
        rubric={"test": 1},
    )

    sample2 = BenchmarkSample(
        id="archbench_002",
        category="messaging",
        difficulty="medium",
        scenario="Unrelated messaging scenario text with kafka streams and partitioning.",
        question="How to partition topics?",
        rubric={"test": 1},
    )

    dataset = BenchmarkDataset([sample1, sample2])
    res = check_benchmark_against_corpus(dataset, corpus_dir, ngram_overlap_threshold=0.30)

    assert res.total_scenarios == 2
    assert res.contaminated_scenarios == 1
    assert res.contamination_rate == 0.50
    assert len(res.flagged_items) == 1
    assert res.flagged_items[0]["sample_id"] == "archbench_001"


def test_benchmark_side_containment_detects_embedding_in_long_document(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "curated"
    corpus_dir.mkdir()
    scenario = (
        "A payment service needs durable ledger writes, idempotent retry handling, "
        "ordered event processing, clear ownership boundaries, observable failures, "
        "and safe recovery after dependent systems become temporarily unavailable."
    )
    question = "Which architecture protects reconciliation correctness while preserving operational simplicity?"
    long_text = f"{scenario} incidental context words {question} " + "background " * 150
    from architectai_pretraining.io import write_dict_jsonl

    write_dict_jsonl([{"id": "embedded", "text": long_text}], corpus_dir / "train.jsonl")
    dataset = BenchmarkDataset(
        [
            BenchmarkSample(
                id="embedded-sample",
                category="architecture_choice",
                difficulty="medium",
                scenario=scenario,
                question=question,
            )
        ]
    )
    result = check_benchmark_against_corpus(dataset, corpus_dir)

    assert result.contaminated_scenarios == 1
    assert result.flagged_items[0]["trigger_reason"] == "high_benchmark_containment"
    assert result.flagged_items[0]["benchmark_ngram_containment"] >= 0.8

# Benchmark test_benchmark_contamination.py test update
