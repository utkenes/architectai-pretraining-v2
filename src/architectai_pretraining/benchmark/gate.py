"""Behavioral regression gate for base-versus-finetuned benchmark runs.

This is intentionally separate from generation knobs.  It detects degradation in
the emitted text and rejects a candidate adapter when it is worse than the base
model under the *same*, explicit chat-template configuration.
"""

import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from architectai_pretraining.benchmark.dataset import BenchmarkDataset
from architectai_pretraining.benchmark.models import EvaluationResult, InferenceConfig, RawOutput
from architectai_pretraining.benchmark.prompts import format_benchmark_prompt


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _has_repetition(text: str) -> bool:
    """Detect repeated non-trivial spans, without penalizing ordinary terminology."""
    tokens = _words(text)
    if len(tokens) < 16:
        return False
    spans = [tuple(tokens[index : index + 8]) for index in range(len(tokens) - 7)]
    return any(spans.count(span) >= 2 for span in set(spans))


def _has_prompt_echo(response: str, prompt: str) -> bool:
    """Flag long contiguous prompt copies that occupy a material response share."""
    response_tokens, prompt_tokens = _words(response), _words(prompt)
    if len(response_tokens) < 16 or len(prompt_tokens) < 12:
        return False
    longest = 0
    for response_start in range(len(response_tokens)):
        for prompt_start in range(len(prompt_tokens)):
            length = 0
            while (
                response_start + length < len(response_tokens)
                and prompt_start + length < len(prompt_tokens)
                and response_tokens[response_start + length] == prompt_tokens[prompt_start + length]
            ):
                length += 1
            longest = max(longest, length)
    return longest >= 12 and longest / len(response_tokens) >= 0.35


@dataclass(frozen=True)
class BehaviorSummary:
    total: int
    repetition_rate: float
    prompt_echo_rate: float
    answer_failure_rate: float
    mean_proxy_score: float


@dataclass(frozen=True)
class QualityGateResult:
    passed: bool
    base: BehaviorSummary
    finetuned: BehaviorSummary
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_behavior(
    prompts: dict[str, str], raw_outputs: list[RawOutput], evaluations: list[EvaluationResult]
) -> BehaviorSummary:
    """Summarize output health for one model run over the frozen benchmark."""
    outputs = {output.sample_id: output for output in raw_outputs}
    score_by_id = {evaluation.sample_id: evaluation.overall_proxy_score for evaluation in evaluations}
    if set(outputs) != set(prompts) or set(score_by_id) != set(prompts):
        raise ValueError("Behavior gate requires complete, matching benchmark outputs and evaluations.")
    repetition = 0
    echo = 0
    answer_failure = 0
    for sample_id, prompt in prompts.items():
        response = outputs[sample_id].raw_response
        repeats = _has_repetition(response)
        echoes = _has_prompt_echo(response, prompt)
        repetition += int(repeats)
        echo += int(echoes)
        answer_failure += int(repeats or echoes or len(_words(response)) < 12)
    total = len(prompts)
    return BehaviorSummary(
        total=total,
        repetition_rate=repetition / total,
        prompt_echo_rate=echo / total,
        answer_failure_rate=answer_failure / total,
        mean_proxy_score=sum(score_by_id.values()) / total,
    )


def compare_base_to_finetuned(
    base: BehaviorSummary,
    finetuned: BehaviorSummary,
    max_rate_regression: float = 0.05,
    max_proxy_score_drop: float = 0.10,
) -> QualityGateResult:
    """Reject behavioral regressions; one failed case is material on a 12-case set."""
    failures: list[str] = []
    for metric in ("repetition_rate", "prompt_echo_rate", "answer_failure_rate"):
        base_value = getattr(base, metric)
        finetuned_value = getattr(finetuned, metric)
        if base_value > max_rate_regression:
            failures.append(f"Base run is unhealthy: {metric}={base_value:.2%}.")
        if finetuned_value > base_value + max_rate_regression:
            failures.append(
                f"Finetuned {metric} regressed from {base_value:.2%} to {finetuned_value:.2%}."
            )
    if finetuned.mean_proxy_score < base.mean_proxy_score - max_proxy_score_drop:
        failures.append(
            "Finetuned instruction-following proxy score regressed "
            f"from {base.mean_proxy_score:.3f} to {finetuned.mean_proxy_score:.3f}."
        )
    return QualityGateResult(not failures, base, finetuned, failures)


def write_quality_gate(path: str | Path, result: QualityGateResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return target


def run_base_vs_finetuned_gate(
    dataset: BenchmarkDataset,
    base_config: InferenceConfig,
    adapter_path: str | Path,
    output_dir: str | Path,
) -> QualityGateResult:
    """Run the exact frozen benchmark for base and adapter, then enforce the gate."""
    if base_config.adapter_path:
        raise ValueError("Base gate configuration must not include an adapter.")
    from architectai_pretraining.benchmark.runner import BenchmarkRunner

    root = Path(output_dir)
    base_runner = BenchmarkRunner(dataset, base_config, root / "base", use_mock=False)
    try:
        _, base_evaluations = base_runner.run()
        base_outputs = list(base_runner._load_completed_raw_outputs().values())
    finally:
        base_runner.release_model()

    finetuned_config = replace(base_config, adapter_path=str(adapter_path))
    finetuned_runner = BenchmarkRunner(dataset, finetuned_config, root / "finetuned", use_mock=False)
    try:
        _, finetuned_evaluations = finetuned_runner.run()
        finetuned_outputs = list(finetuned_runner._load_completed_raw_outputs().values())
    finally:
        finetuned_runner.release_model()
    prompts = {sample.id: format_benchmark_prompt(sample) for sample in dataset.samples}
    result = compare_base_to_finetuned(
        summarize_behavior(prompts, base_outputs, base_evaluations),
        summarize_behavior(prompts, finetuned_outputs, finetuned_evaluations),
    )
    write_quality_gate(root / "quality_gate.json", result)
    if not result.passed:
        raise RuntimeError("DAPT behavior gate failed: " + " ".join(result.failures))
    return result
