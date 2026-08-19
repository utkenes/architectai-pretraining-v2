"""Data models and schemas for ArchitectAI Stage 4 Benchmark and Evaluation Harness."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class BenchmarkSample:
    """Represents a single architectural evaluation scenario."""

    id: str
    category: str
    difficulty: str  # "easy", "medium", "hard"
    scenario: str
    question: str
    facts: list[str] = field(default_factory=list)
    expected_considerations: list[str] = field(default_factory=list)
    rubric: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "difficulty": self.difficulty,
            "scenario": self.scenario,
            "question": self.question,
            "facts": self.facts,
            "expected_considerations": self.expected_considerations,
            "rubric": self.rubric,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkSample":
        return cls(
            id=data["id"],
            category=data.get("category", "architecture_choice"),
            difficulty=data.get("difficulty", "medium"),
            scenario=data.get("scenario", ""),
            question=data.get("question", ""),
            facts=data.get("facts", []),
            expected_considerations=data.get("expected_considerations", []),
            rubric=data.get("rubric", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class InferenceConfig:
    """Configuration details for model inference execution."""

    model_identifier: str = "Qwen/Qwen3-8B"
    revision: str = "main"
    tokenizer_identifier: str = "Qwen/Qwen3-8B"
    quantization: str = "none"  # "none", "4bit", "8bit"
    dtype: str = "bfloat16"  # "bfloat16", "float16", "float32"
    device: str = "cuda"  # "cuda", "cpu"
    temperature: float = 0.0
    do_sample: bool = False
    max_new_tokens: int = 768
    enable_thinking: bool = False
    adapter_path: str | None = None
    checkpoint_metadata_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_identifier": self.model_identifier,
            "revision": self.revision,
            "tokenizer_identifier": self.tokenizer_identifier,
            "quantization": self.quantization,
            "dtype": self.dtype,
            "device": self.device,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            "enable_thinking": self.enable_thinking,
            "adapter_path": self.adapter_path,
            "checkpoint_metadata_path": self.checkpoint_metadata_path,
        }


@dataclass
class RawOutput:
    """Preserved raw model generation output."""

    sample_id: str
    model_identifier: str
    prompt_hash: str
    raw_response: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    is_mock: bool = False
    inference_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "model_identifier": self.model_identifier,
            "prompt_hash": self.prompt_hash,
            "raw_response": self.raw_response,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_seconds": self.latency_seconds,
            "timestamp": self.timestamp,
            "is_mock": self.is_mock,
            "inference_fingerprint": self.inference_fingerprint,
        }


@dataclass
class EvaluationResult:
    """Result of deterministic/heuristic rubric evaluation for a single sample."""

    sample_id: str
    category: str
    difficulty: str
    scores: dict[str, float]  # Heuristic proxy dimension scores (0.0 - 1.0)
    overall_proxy_score: float
    unsupported_claim_rate: float
    unsupported_claims: list[str] = field(default_factory=list)
    revisit_conditions_identified: list[str] = field(default_factory=list)
    missing_info_acknowledged: bool = False
    is_mock_eval: bool = False
    evaluator_type: str = "deterministic_proxy_v1"  # Explicitly labeled as proxy/heuristic

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "scores": self.scores,
            "overall_proxy_score": self.overall_proxy_score,
            "unsupported_claim_rate": self.unsupported_claim_rate,
            "unsupported_claims": self.unsupported_claims,
            "revisit_conditions_identified": self.revisit_conditions_identified,
            "missing_info_acknowledged": self.missing_info_acknowledged,
            "is_mock_eval": self.is_mock_eval,
            "evaluator_type": self.evaluator_type,
        }


@dataclass
class ContaminationResult:
    """Result of benchmark vs training corpus contamination analysis."""

    total_scenarios: int
    contaminated_scenarios: int
    contamination_rate: float
    flagged_items: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_scenarios": self.total_scenarios,
            "contaminated_scenarios": self.contaminated_scenarios,
            "contamination_rate": round(self.contamination_rate, 4),
            "flagged_items": self.flagged_items,
        }


@dataclass
class RubricCriteria:
    """Dimensions evaluated in the architecture rubric."""

    driver_identification: float = 0.0
    nfr_coverage: float = 0.0
    architecture_choice_quality: float = 0.0
    tradeoff_quality: float = 0.0
    alternative_analysis: float = 0.0
    risk_identification: float = 0.0
    reliability_reasoning: float = 0.0
    scalability_reasoning: float = 0.0
    operational_complexity_awareness: float = 0.0
    cost_awareness: float = 0.0
    unsupported_claim_avoidance: float = 1.0
    revisit_conditions: float = 0.0
    clarification_awareness: float = 0.0
    reasoning_completeness: float = 0.0


@dataclass
class BenchmarkResultManifest:
    """Manifest summarizing baseline benchmark execution."""

    benchmark_version: str = "architectai-bench-v1"
    benchmark_fingerprint: str = ""
    model_identifier: str = "Qwen/Qwen3-8B"
    model_revision: str = "main"
    tokenizer_identifier: str = "Qwen/Qwen3-8B"
    inference_config: dict[str, Any] = field(default_factory=dict)
    git_commit: str = ""
    execution_environment: str = ""
    scenario_count: int = 0
    completed_count: int = 0
    result_fingerprint: str = ""
    is_mock_run: bool = False
    ready_for_stage_5: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_version": self.benchmark_version,
            "benchmark_fingerprint": self.benchmark_fingerprint,
            "model_identifier": self.model_identifier,
            "model_revision": self.model_revision,
            "tokenizer_identifier": self.tokenizer_identifier,
            "inference_config": self.inference_config,
            "git_commit": self.git_commit,
            "execution_environment": self.execution_environment,
            "scenario_count": self.scenario_count,
            "completed_count": self.completed_count,
            "result_fingerprint": self.result_fingerprint,
            "is_mock_run": self.is_mock_run,
            "ready_for_stage_5": self.ready_for_stage_5,
        }


@dataclass
class BaselineReport:
    """Summary metrics of baseline run."""

    overall_proxy_score: float
    category_scores: dict[str, float]
    difficulty_scores: dict[str, float]
    dimension_scores: dict[str, float]
    unsupported_claim_rate: float
    revisit_condition_rate: float
    clarification_awareness_rate: float
    total_samples: int
    is_mock_run: bool
    ready_for_stage_5: bool

# Benchmark models.py module update
