# Post-training Qwen3 Quality Gate

## What this protects

The Stage 5 DAPT corpus is deliberately raw architecture documentation, not a
chat/instruction corpus. Continuing next-token training only on this data can
improve terminology while eroding the Qwen3 assistant distribution, especially
with full-parameter updates, broad LoRA targets, or excessive update counts.
This is a catastrophic-forgetting/alignment risk, not a claim that every poor
response is caused by training.

There was also an independent benchmark issue: Qwen3 enables thinking by
default, while the old benchmark used greedy decoding. The Qwen3 model card
explicitly warns not to use greedy decoding in thinking mode because it can
cause degraded quality and endless repetition. The benchmark now passes
`enable_thinking=False` explicitly, making its greedy run deterministic and
comparable. See the [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-8B).

## Required promotion command

Run the base and adapter on the exact frozen v1 benchmark, with separate output
directories and the same explicit template/generation configuration:

```powershell
architectai-pretraining benchmark gate `
  --dataset data/benchmark/architectai_v1.jsonl `
  --results-dir data/benchmark/results/dapt_gate `
  --model Qwen/Qwen3-8B `
  --tokenizer Qwen/Qwen3-8B `
  --adapter-path outputs/dapt-smoke/checkpoint-20/model
```

The command refuses mock inference, refuses adapters without
`adapter_config.json` and `checkpoint_metadata.json`, and confirms the base
model, revision, and tokenizer identity before it calls PEFT. It writes
`quality_gate.json` and exits non-zero if the candidate is unsafe.

The default smoke configuration preserves this 80-case v1 gate. To add the
separate frozen 12-case diagnostic set, set
`evaluation.diagnostic_dataset_path` in `configs/dapt.yaml`; it runs only after
the v1 gate and writes results below `dapt_gate/diagnostic`. It never replaces or
modifies benchmark v1. Both runs use 4-bit float16 inference and release the
base model before loading the adapter so they fit QLoRA-class Colab GPUs.

The gate rejects a run when either model has even one failure case in the
12-case benchmark (which is already above the 5% threshold), or the finetuned model increases repetition, prompt
echo, or answer-failure rate by more than 5 percentage points. It also rejects a
drop greater than 0.10 in the deterministic instruction-following proxy. These
are safeguards and not substitutes for human review.

## Conservative DAPT defaults

`configs/dapt.yaml` now defaults to QLoRA, rank 8, alpha 16, and only Qwen3
`q_proj`/`v_proj` targets with a `5e-6` learning rate, 10% warmup, and 0.01
weight decay. The smoke runner counts `max_steps` as optimizer updates (not
microbatches), applies real accumulation, and applies warmup/linear decay.

For any longer run, retain a held-out base-vs-adapter gate at each checkpoint.
Before increasing the adapter scope or learning rate, add a small, licensed
instruction/general-text replay mixture; raw-document DAPT alone is not an
alignment-preservation method. Do not mask a failed gate by increasing
`repetition_penalty`: reproduce the same failure with the explicit benchmark
configuration and fix the training recipe or checkpoint selection.
