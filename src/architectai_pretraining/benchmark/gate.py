"""Behavioral regression gate for base-versus-finetuned benchmark runs.

This is intentionally separate from generation knobs.  It detects degradation in
the emitted text and rejects a candidate adapter when it is worse than the base
model under the *same*, explicit chat-template configuration.
"""

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from architectai_pretraining.benchmark.dataset import BenchmarkDataset
from architectai_pretraining.benchmark.models import EvaluationResult, InferenceConfig, RawOutput
from architectai_pretraining.benchmark.prompts import format_benchmark_prompt


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _has_repetition(text: str) -> bool:
    """Detect local or persistent repetition loops without penalizing comparisons.

    A phrase used twice in distinct architecture options is ordinary explanatory
    reuse.  A loop repeats the same span locally, or keeps returning to it often.
    """
    tokens = _words(text)
    if len(tokens) < 16:
        return False
    span_width = 8
    local_loop_gap = span_width * 2
    positions: dict[tuple[str, ...], list[int]] = {}
    for index in range(len(tokens) - span_width + 1):
        span = tuple(tokens[index : index + span_width])
        positions.setdefault(span, []).append(index)
    for occurrences in positions.values():
        if len(occurrences) >= 3:
            return True
        if len(occurrences) == 2 and occurrences[1] - occurrences[0] <= local_loop_gap:
            return True
    return False


REPETITION_DETECTOR_V3 = "v3_local_cluster_density"


def _has_repetition_v3(text: str) -> bool:
    """Detect degenerate local loops while allowing long-form architectural reuse.

    Version 2 treated any three appearances of an eight-token phrase as a loop.
    In architecture answers, a decision threshold or option name can legitimately
    appear in an option analysis, a recommendation, and revisit conditions.  V3
    therefore requires evidence that the repeated text is either concentrated in
    a local loop or consumes a material share of a short response.
    """
    tokens = _words(text)
    if len(tokens) < 16:
        return False
    span_width = 8
    local_loop_gap = span_width * 2
    local_cluster_window = span_width * 4
    density_threshold = 0.12
    positions: dict[tuple[str, ...], list[int]] = {}
    for index in range(len(tokens) - span_width + 1):
        span = tuple(tokens[index : index + span_width])
        positions.setdefault(span, []).append(index)
    for occurrences in positions.values():
        repeated_coverage = len(occurrences) * span_width / len(tokens)
        if len(occurrences) >= 2:
            has_close_pair = any(
                later - earlier <= local_loop_gap
                for earlier, later in zip(occurrences, occurrences[1:], strict=False)
            )
            if has_close_pair and repeated_coverage >= density_threshold:
                return True
        if len(occurrences) >= 3:
            has_local_cluster = any(
                occurrences[index + 2] - occurrences[index] <= local_cluster_window
                for index in range(len(occurrences) - 2)
            )
            if has_local_cluster:
                return True
    return False


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
    truncation_rate: float
    mean_proxy_score: float


@dataclass(frozen=True)
class QualityGateResult:
    passed: bool
    base: BehaviorSummary
    finetuned: BehaviorSummary
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _summarize_behavior(
    prompts: dict[str, str],
    raw_outputs: list[RawOutput],
    evaluations: list[EvaluationResult],
    max_new_tokens: int | None = None,
    repetition_detector: Callable[[str], bool] = _has_repetition,
) -> BehaviorSummary:
    """Summarize output health with an explicitly selected repetition detector."""
    outputs = {output.sample_id: output for output in raw_outputs}
    score_by_id = {evaluation.sample_id: evaluation.overall_proxy_score for evaluation in evaluations}
    if set(outputs) != set(prompts) or set(score_by_id) != set(prompts):
        raise ValueError("Behavior gate requires complete, matching benchmark outputs and evaluations.")
    repetition = 0
    echo = 0
    answer_failure = 0
    truncated = 0
    for sample_id, prompt in prompts.items():
        response = outputs[sample_id].raw_response
        repeats = repetition_detector(response)
        echoes = _has_prompt_echo(response, prompt)
        repetition += int(repeats)
        echo += int(echoes)
        answer_failure += int(repeats or echoes or len(_words(response)) < 12)
        truncated += int(max_new_tokens is not None and outputs[sample_id].output_tokens >= max_new_tokens)
    total = len(prompts)
    return BehaviorSummary(
        total=total,
        repetition_rate=repetition / total,
        prompt_echo_rate=echo / total,
        answer_failure_rate=answer_failure / total,
        truncation_rate=truncated / total,
        mean_proxy_score=sum(score_by_id.values()) / total,
    )


def summarize_behavior(
    prompts: dict[str, str],
    raw_outputs: list[RawOutput],
    evaluations: list[EvaluationResult],
    max_new_tokens: int | None = None,
) -> BehaviorSummary:
    """Summarize behavior using the original V2 detector for gate compatibility."""
    return _summarize_behavior(prompts, raw_outputs, evaluations, max_new_tokens, _has_repetition)


def summarize_behavior_v3(
    prompts: dict[str, str],
    raw_outputs: list[RawOutput],
    evaluations: list[EvaluationResult],
    max_new_tokens: int | None = None,
) -> BehaviorSummary:
    """Summarize behavior with the versioned V3 local-loop detector."""
    return _summarize_behavior(prompts, raw_outputs, evaluations, max_new_tokens, _has_repetition_v3)


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
        summarize_behavior(
            prompts, base_outputs, base_evaluations, max_new_tokens=base_config.max_new_tokens
        ),
        summarize_behavior(
            prompts, finetuned_outputs, finetuned_evaluations,
            max_new_tokens=finetuned_config.max_new_tokens,
        ),
    )
    write_quality_gate(root / "quality_gate.json", result)
    if not result.passed:
        raise RuntimeError("DAPT behavior gate failed: " + " ".join(result.failures))
    return result
