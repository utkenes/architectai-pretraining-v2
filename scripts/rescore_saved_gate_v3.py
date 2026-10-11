#!/usr/bin/env python3
"""Re-score saved benchmark outputs with the versioned V3 repetition detector.

This utility never runs inference and never writes to its source result tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from architectai_pretraining.benchmark.dataset import load_benchmark_dataset
from architectai_pretraining.benchmark.gate import (
    REPETITION_DETECTOR_V3,
    compare_base_to_finetuned,
    summarize_behavior_v3,
)
from architectai_pretraining.benchmark.models import EvaluationResult, RawOutput
from architectai_pretraining.benchmark.prompts import format_benchmark_prompt


def _load_raw_outputs(path: Path) -> list[RawOutput]:
    return [RawOutput(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_evaluations(path: Path) -> list[EvaluationResult]:
    return [EvaluationResult(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    dataset = load_benchmark_dataset(args.dataset)
    prompts = {sample.id: format_benchmark_prompt(sample) for sample in dataset.samples}
    summaries = {}
    for run in ("base", "finetuned"):
        source = args.source_results / run
        raw_outputs = _load_raw_outputs(source / "raw_outputs.jsonl")
        evaluations = _load_evaluations(source / "evaluations.jsonl")
        summaries[run] = summarize_behavior_v3(
            prompts, raw_outputs, evaluations, max_new_tokens=args.max_new_tokens
        )
    result = compare_base_to_finetuned(summaries["base"], summaries["finetuned"])
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "quality_gate_v3.json"
    target.write_text(
        json.dumps(
            {
                "source_results": str(args.source_results),
                "detector_version": REPETITION_DETECTOR_V3,
                "gate_contract": {"max_rate_regression": 0.05, "max_proxy_score_drop": 0.10},
                **result.to_dict(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(target)


if __name__ == "__main__":
    main()
