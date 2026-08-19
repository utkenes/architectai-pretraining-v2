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


def is_benchmark_path_excluded(path: str | Path) -> bool:
    """Verifies whether a file path falls under benchmark data directories that MUST be excluded from ingestion."""
    norm_path = str(Path(path)).replace("\\", "/").lower()
    return "data/benchmark" in norm_path or "benchmark/" in norm_path


def check_benchmark_against_corpus(
    dataset: BenchmarkDataset,
    corpus_dir: str | Path,
    ngram_overlap_threshold: float = 0.50,
) -> Any:
    """Compares benchmark scenarios against raw/curated training corpus JSONL files.

    Detects exact text matches and significant n-gram overlaps.
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
    sample_ngrams = {
        s.id: _get_ngrams(f"{s.scenario} {s.question}") for s in dataset.samples
    }

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

            doc_ngrams = _get_ngrams(doc_text)
            if not doc_ngrams:
                continue

            for sample in dataset.samples:
                s_ngrams = sample_ngrams[sample.id]
                if not s_ngrams:
                    continue

                intersection = len(s_ngrams & doc_ngrams)
                jaccard = intersection / max(1, len(s_ngrams | doc_ngrams))

                if jaccard >= ngram_overlap_threshold:
                    contaminated_ids.add(sample.id)
                    flagged_items.append(
                        {
                            "sample_id": sample.id,
                            "corpus_file": str(jsonl_file),
                            "corpus_doc_id": doc_id,
                            "ngram_jaccard": round(jaccard, 4),
                            "reason": f"High n-gram overlap ({jaccard:.2%}) with training document {doc_id}",
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
