"""Contamination detection module ensuring benchmark scenarios do not leak into training data."""

import logging
from pathlib import Path
from typing import Any

from architectai_pretraining.benchmark.dataset import BenchmarkDataset

logger = logging.getLogger(__name__)


def _get_ngrams(text: str, n: int = 4) -> set[str]:
    words = text.lower().split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def is_benchmark_path_excluded(path: str | Path) -> bool:
    """Verifies whether a file path falls under benchmark data directories that MUST be excluded from ingestion."""
    norm_path = str(Path(path)).replace("\\", "/").lower()
    return "data/benchmark" in norm_path or "benchmark/" in norm_path


def check_benchmark_against_corpus(
    dataset: BenchmarkDataset,
    corpus_dir: str | Path,
    ngram_overlap_threshold: float = 0.50,
    benchmark_containment_threshold: float = 0.80,
    minimum_benchmark_ngrams: int = 5,
    minimum_matching_ngrams: int = 4,
) -> Any:
    """Compares benchmark scenarios against raw/curated training corpus JSONL files.

    Detects exact benchmark reuse, Jaccard overlap, and high benchmark-side
    containment. Containment is separately guarded for short samples so one
    generic phrase cannot contaminate a scenario.
    """

    from architectai_pretraining.benchmark.models import ContaminationResult

    corpus_path = Path(corpus_dir)
    jsonl_files: list[Path] = []
    if corpus_path.is_file() and corpus_path.suffix == ".jsonl":
        jsonl_files.append(corpus_path)
    elif corpus_path.is_dir():
        jsonl_files.extend(list(corpus_path.rglob("*.jsonl")))

    if not jsonl_files:
        logger.warning("No JSONL files found in corpus directory %s for contamination check.", corpus_dir)
        return ContaminationResult(
            total_scenarios=len(dataset.samples),
            contaminated_scenarios=0,
            contamination_rate=0.0,
            flagged_items=[],
        )

    flagged_items: list[dict[str, Any]] = []
    contaminated_ids: set[str] = set()

    # Pre-index benchmark scenario n-grams
    sample_texts = {s.id: _normalized(f"{s.scenario} {s.question}") for s in dataset.samples}
    sample_ngrams = {sample.id: _get_ngrams(sample_texts[sample.id]) for sample in dataset.samples}

    from architectai_pretraining.io import iter_dict_jsonl

    for jsonl_file in jsonl_files:
        # Ignore benchmark result files themselves
        if is_benchmark_path_excluded(jsonl_file):
            continue

        for doc_dict in iter_dict_jsonl(jsonl_file):
            doc_id = str(doc_dict.get("id", "unknown"))
            doc_text = str(doc_dict.get("text", ""))
            if not doc_text:
                continue
            normalized_doc = _normalized(doc_text)
            doc_ngrams = _get_ngrams(normalized_doc)
            if not doc_ngrams:
                continue

            for sample in dataset.samples:
                s_ngrams = sample_ngrams[sample.id]
                if not s_ngrams:
                    continue

                intersection = len(s_ngrams & doc_ngrams)
                jaccard = intersection / max(1, len(s_ngrams | doc_ngrams))
                containment = intersection / len(s_ngrams)
                exact_match = sample_texts[sample.id] in normalized_doc
                high_containment = (
                    len(s_ngrams) >= minimum_benchmark_ngrams
                    and intersection >= minimum_matching_ngrams
                    and containment >= benchmark_containment_threshold
                )
                if exact_match or jaccard >= ngram_overlap_threshold or high_containment:
                    trigger_reason = (
                        "exact_benchmark_text"
                        if exact_match
                        else "high_benchmark_containment"
                        if high_containment
                        else "high_ngram_jaccard"
                    )
                    contaminated_ids.add(sample.id)
                    flagged_items.append(
                        {
                            "sample_id": sample.id,
                            "corpus_file": str(jsonl_file),
                            "corpus_doc_id": doc_id,
                            "ngram_jaccard": round(jaccard, 4),
                            "benchmark_ngram_containment": round(containment, 4),
                            "matching_ngram_count": intersection,
                            "benchmark_ngram_count": len(s_ngrams),
                            "trigger_reason": trigger_reason,
                            "reason": f"{trigger_reason} with training document {doc_id}",
                        }
                    )

    rate = len(contaminated_ids) / max(1, len(dataset.samples))
    return ContaminationResult(
        total_scenarios=len(dataset.samples),
        contaminated_scenarios=len(contaminated_ids),
        contamination_rate=rate,
        flagged_items=flagged_items,
    )

# Contamination checker module
