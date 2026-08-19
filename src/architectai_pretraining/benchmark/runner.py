"""Resumable benchmark execution engine supporting real HF inference & mock validation modes."""

import hashlib
import json
import logging
import time
from gc import collect
from pathlib import Path
from typing import Any

from architectai_pretraining.benchmark.dataset import BenchmarkDataset
from architectai_pretraining.benchmark.models import (
    BenchmarkResultManifest,
    BenchmarkSample,
    EvaluationResult,
    InferenceConfig,
    RawOutput,
)
from architectai_pretraining.benchmark.prompts import FROZEN_SYSTEM_PROMPT, format_benchmark_prompt
from architectai_pretraining.benchmark.scoring import (
    DeterministicRubricEvaluator,
    export_human_review,
)

logger = logging.getLogger(__name__)


def _adapter_fingerprint(adapter_path: str | None) -> str:
    """Fingerprint adapter contents so resumable results never mix checkpoints."""
    if not adapter_path:
        return ""
    root = Path(adapter_path)
    if not root.exists():
        return "missing"
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


class BenchmarkRunner:
    """Orchestrates base Qwen model inference, raw output logging, and evaluation."""

    def __init__(
        self,
        dataset: BenchmarkDataset,
        config: InferenceConfig | None = None,
        results_dir: str | Path = "data/benchmark/results/baseline",
        use_mock: bool = False,
    ) -> None:
        self.dataset = dataset
        self.config = config or InferenceConfig()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.use_mock = use_mock
        fingerprint_input = self.config.to_dict() | {"adapter_contents": _adapter_fingerprint(self.config.adapter_path)}
        self._inference_fingerprint = hashlib.sha256(json.dumps(fingerprint_input, sort_keys=True).encode("utf-8")).hexdigest()

        self.raw_outputs_file = self.results_dir / "raw_outputs.jsonl"
        self.evaluations_file = self.results_dir / "evaluations.jsonl"
        self.manifest_file = self.results_dir / "baseline_manifest.json"
        self.human_review_file = self.results_dir / "human_review.jsonl"

        self.model: Any = None
        self.tokenizer: Any = None

    def release_model(self) -> None:
        """Release model references before the gate loads its next 8B model."""
        self.model = None
        self.tokenizer = None
        collect()
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _load_completed_raw_outputs(self) -> dict[str, RawOutput]:
        """Loads previously saved raw outputs to allow resumable execution."""
        completed: dict[str, RawOutput] = {}
        if not self.raw_outputs_file.exists():
            return completed

        from architectai_pretraining.io import iter_dict_jsonl

        for item in iter_dict_jsonl(self.raw_outputs_file):
            raw = RawOutput(
                sample_id=str(item["sample_id"]),
                model_identifier=str(item["model_identifier"]),
                prompt_hash=str(item["prompt_hash"]),
                raw_response=str(item["raw_response"]),
                input_tokens=int(item.get("input_tokens", 0)),
                output_tokens=int(item.get("output_tokens", 0)),
                latency_seconds=float(item.get("latency_seconds", 0.0)),
                timestamp=str(item.get("timestamp", "")),
                is_mock=bool(item.get("is_mock", False)),
                inference_fingerprint=str(item.get("inference_fingerprint", "")),
            )
            if raw.inference_fingerprint == self._inference_fingerprint:
                completed[raw.sample_id] = raw

        return completed

    def _init_real_model(self) -> None:
        """Initializes HuggingFace Transformer model & tokenizer with quantization options."""
        if self.model is not None:
            return

        if self.config.enable_thinking and not self.config.do_sample:
            raise ValueError(
                "Qwen3 thinking-mode evaluation must use sampling; use "
                "enable_thinking=False for deterministic benchmark runs."
            )

        logger.info("Initializing HuggingFace model %s...", self.config.model_identifier)
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as err:
            raise RuntimeError(
                "PyTorch and Transformers are required for real model execution. "
                "Install them or pass use_mock=True for test validation."
            ) from err

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_identifier,
            revision=self.config.revision,
            trust_remote_code=True,
        )

        # Quantization Config
        quant_config = None
        if self.config.quantization == "4bit":
            quant_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.dtype, torch.bfloat16),
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)  # type: ignore[no-untyped-call]

        dtype_val = getattr(torch, self.config.dtype, torch.bfloat16) if hasattr(torch, self.config.dtype) else torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_identifier,
            revision=self.config.revision,
            quantization_config=quant_config,
            torch_dtype=dtype_val,
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.config.adapter_path:
            self._load_verified_adapter()

    def _load_verified_adapter(self) -> None:
        """Load a PEFT adapter only after proving its base-model identity.

        Evaluating an adapter as if it were a standalone checkpoint silently gives
        meaningless results.  Stage 5 checkpoints always include metadata, so a
        missing or mismatched metadata file is an evaluation error, not a warning.
        """
        adapter_path = Path(self.config.adapter_path or "")
        adapter_config_path = adapter_path / "adapter_config.json"
        metadata_path = Path(self.config.checkpoint_metadata_path or adapter_path.parent / "checkpoint_metadata.json")
        if not adapter_config_path.exists() or not metadata_path.exists():
            raise ValueError(
                "Finetuned evaluation requires adapter_config.json and checkpoint_metadata.json."
            )
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_base = self.config.model_identifier
        if adapter_config.get("base_model_name_or_path") != expected_base:
            raise ValueError("Adapter base model does not match benchmark model_identifier.")
        if metadata.get("base_model_identifier") != expected_base:
            raise ValueError("Checkpoint metadata base model does not match benchmark model_identifier.")
        if metadata.get("base_revision") != self.config.revision:
            raise ValueError("Checkpoint metadata revision does not match benchmark revision.")
        if metadata.get("tokenizer_identifier") != self.config.tokenizer_identifier:
            raise ValueError("Checkpoint metadata tokenizer does not match benchmark tokenizer.")
        try:
            from peft import PeftModel  # type: ignore[import-not-found]
        except ImportError as err:
            raise RuntimeError("PEFT is required to evaluate a finetuned adapter.") from err
        self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)

    def _generate_mock_response(self, sample: BenchmarkSample) -> str:
        """Generates realistic structured mock responses for fast unit test verification."""
        scenario_lower = sample.scenario.lower()
        tradeoffs = "Trade-offs between availability and consistency must be carefully evaluated based on scenario facts."

        if "microservices" in scenario_lower:
            tradeoffs += " Microservices add network complexity, operational overhead, and distributed transaction challenges."

        revisit = "Revisit this architectural decision if p95 latency exceeds 500ms or queue lag accumulates beyond 10,000 messages."

        missing_info = ""
        if sample.metadata.get("has_missing_info"):
            missing_info = " Missing Information: Exact RTO/RPO SLA targets and total cloud budget are unspecified and must be clarified."

        return f"""### Architectural Analysis & Recommendations ({sample.id})

1. Architectural Drivers & Constraints:
- Traffic volume: High concurrency demands scalable NFR consideration.
- Data consistency: Operational requirements require clear state persistence boundaries.

2. Architecture Recommendation:
Based on scenario facts, a defensive architecture with clear domain boundaries is recommended. {tradeoffs}

3. Risk & Reliability Considerations:
Implement circuit breakers, retries with exponential backoff, and dead-letter queues to prevent cascading failures.

4. Revisit Conditions:
{revisit}
{missing_info}"""

    def _generate_single_response(self, sample: BenchmarkSample) -> tuple[str, int, int, float]:
        """Generates response for a single scenario."""
        prompt = format_benchmark_prompt(sample)
        start_time = time.time()

        if self.use_mock:
            response_text = self._generate_mock_response(sample)
            latency = round(time.time() - start_time, 4)
            in_tokens = len(prompt.split())
            out_tokens = len(response_text.split())
            return response_text, in_tokens, out_tokens, latency

        self._init_real_model()
        messages = [
            {"role": "system", "content": FROZEN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,
        )

        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)
        in_tokens = inputs.input_ids.shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        latency = round(time.time() - start_time, 4)
        out_tokens = outputs.shape[1] - in_tokens
        gen_tokens = outputs[0][in_tokens:]
        response_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        return response_text, in_tokens, out_tokens, latency

    def run(self) -> tuple[BenchmarkResultManifest, list[EvaluationResult]]:
        """Runs resumable benchmark inference and evaluates baseline outputs."""
        logger.info(
            "Starting benchmark execution for dataset '%s' (%d scenarios)...",
            self.dataset.fingerprint[:12],
            len(self.dataset.samples),
        )

        completed_raw = self._load_completed_raw_outputs()
        raw_outputs: list[RawOutput] = list(completed_raw.values())
        evaluator = DeterministicRubricEvaluator()
        evaluations: list[EvaluationResult] = []

        # Process each scenario (skipping completed ones for resumability)
        for idx, sample in enumerate(self.dataset.samples):
            if sample.id in completed_raw:
                raw_out = completed_raw[sample.id]
                logger.info("[%d/%d] Using cached raw output for %s", idx + 1, len(self.dataset.samples), sample.id)
            else:
                prompt_text = format_benchmark_prompt(sample)
                prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

                resp_text, in_tok, out_tok, lat = self._generate_single_response(sample)
                raw_out = RawOutput(
                    sample_id=sample.id,
                    model_identifier=self.config.model_identifier,
                    prompt_hash=prompt_hash,
                    raw_response=resp_text,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_seconds=lat,
                    is_mock=self.use_mock,
                    inference_fingerprint=self._inference_fingerprint,
                )
                completed_raw[sample.id] = raw_out
                raw_outputs.append(raw_out)
                from architectai_pretraining.io import write_dict_jsonl

                write_dict_jsonl([r.to_dict() for r in raw_outputs], self.raw_outputs_file)
                logger.info("[%d/%d] Generated raw response for %s", idx + 1, len(self.dataset.samples), sample.id)

            # Evaluate sample
            eval_res = evaluator.evaluate(sample, raw_out)
            evaluations.append(eval_res)

        # Persist evaluations JSONL
        write_dict_jsonl([e.to_dict() for e in evaluations], self.evaluations_file)

        # Export human review JSONL
        export_human_review(self.dataset.samples, raw_outputs, evaluations, self.human_review_file)

        # Compute result fingerprint
        result_hasher = hashlib.sha256()
        for raw in sorted(raw_outputs, key=lambda x: x.sample_id):
            result_hasher.update(raw.raw_response.encode("utf-8"))
        result_fp = result_hasher.hexdigest()

        # User Correction #1 & #2: NEVER set ready_for_stage_5=True from mock runs!
        is_mock_run = self.use_mock or any(r.is_mock for r in raw_outputs)
        ready_for_stage_5 = not is_mock_run and (len(evaluations) == len(self.dataset.samples))

        manifest = BenchmarkResultManifest(
            benchmark_version="architectai-bench-v1",
            benchmark_fingerprint=self.dataset.fingerprint,
            model_identifier=self.config.model_identifier,
            model_revision=self.config.revision,
            tokenizer_identifier=self.config.tokenizer_identifier,
            inference_config=self.config.to_dict(),
            git_commit="",
            execution_environment="Colab/Local",
            scenario_count=len(self.dataset.samples),
            completed_count=len(evaluations),
            result_fingerprint=result_fp,
            is_mock_run=is_mock_run,
            ready_for_stage_5=ready_for_stage_5,
        )

        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        return manifest, evaluations

# Benchmark runner.py module update
