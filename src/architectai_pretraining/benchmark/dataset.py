"""Dataset loader, schema validator, and fingerprint computation for ArchitectAI Benchmark."""

import hashlib
import json
from pathlib import Path

from architectai_pretraining.benchmark.models import BenchmarkSample


def compute_benchmark_fingerprint(samples: list[BenchmarkSample]) -> str:
    """Computes a deterministic SHA-256 fingerprint for frozen benchmark samples."""
    sorted_samples = sorted(samples, key=lambda x: x.id)
    hasher = hashlib.sha256()

    for sample in sorted_samples:
        payload = json.dumps(
            {
                "id": sample.id,
                "category": sample.category,
                "difficulty": sample.difficulty,
                "scenario": sample.scenario,
                "question": sample.question,
                "facts": sample.facts,
                "expected_considerations": sample.expected_considerations,
                "rubric": sample.rubric,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        hasher.update(payload.encode("utf-8"))

    return hasher.hexdigest()


class BenchmarkDataset:
    """Encapsulates frozen benchmark scenarios and distribution statistics."""

    def __init__(self, samples: list[BenchmarkSample], dataset_path: str | Path | None = None) -> None:
        self.samples = samples
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.fingerprint = compute_benchmark_fingerprint(samples)

    def validate(self) -> tuple[bool, list[str]]:
        """Validates schema integrity, non-empty fields, and valid categories/difficulties."""
        errors: list[str] = []
        valid_difficulties = {"easy", "medium", "hard"}
        seen_ids: set[str] = set()

        if not self.samples:
            errors.append("Benchmark dataset contains zero samples.")
            return False, errors

        for idx, sample in enumerate(self.samples):
            prefix = f"Sample #{idx+1} (ID: '{sample.id}')"
            if not sample.id or not isinstance(sample.id, str):
                errors.append(f"{prefix}: Missing or invalid ID.")
            elif sample.id in seen_ids:
                errors.append(f"{prefix}: Duplicate ID found.")
            else:
                seen_ids.add(sample.id)

            if not sample.category or not isinstance(sample.category, str):
                errors.append(f"{prefix}: Missing category.")

            if sample.difficulty not in valid_difficulties:
                errors.append(
                    f"{prefix}: Invalid difficulty '{sample.difficulty}'. Must be one of {valid_difficulties}."
                )

            if not sample.scenario or len(sample.scenario.strip()) < 20:
                errors.append(f"{prefix}: Scenario text is missing or too short.")

            if not sample.question or len(sample.question.strip()) < 5:
                errors.append(f"{prefix}: Question text is missing or too short.")

            if not sample.rubric or not isinstance(sample.rubric, dict):
                errors.append(f"{prefix}: Missing or invalid rubric dictionary.")

        return len(errors) == 0, errors

    def get_category_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for sample in self.samples:
            dist[sample.category] = dist.get(sample.category, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))

    def get_difficulty_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
        for sample in self.samples:
            dist[sample.difficulty] = dist.get(sample.difficulty, 0) + 1
        return dist

    def get_audit_sample(
        self, min_easy: int = 5, min_medium: int = 10, min_hard: int = 10
    ) -> list[BenchmarkSample]:
        """Selects a representative sample across difficulty levels for human audit."""
        by_diff: dict[str, list[BenchmarkSample]] = {"easy": [], "medium": [], "hard": []}
        for s in self.samples:
            if s.difficulty in by_diff:
                by_diff[s.difficulty].append(s)

        selected: list[BenchmarkSample] = []
        selected.extend(by_diff["easy"][:min_easy])
        selected.extend(by_diff["medium"][:min_medium])
        selected.extend(by_diff["hard"][:min_hard])

        return selected


def load_benchmark_dataset(file_path: str | Path) -> BenchmarkDataset:
    """Loads benchmark dataset from JSONL file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset file not found: {path}")

    from architectai_pretraining.io import iter_dict_jsonl

    raw_dicts = iter_dict_jsonl(path)
    samples = [BenchmarkSample.from_dict(d) for d in raw_dicts]
    dataset = BenchmarkDataset(samples, dataset_path=path)
    is_valid, errors = dataset.validate()
    if not is_valid:
        raise ValueError(f"Benchmark dataset validation failed: {'; '.join(errors)}")

    return dataset

# Benchmark dataset module
