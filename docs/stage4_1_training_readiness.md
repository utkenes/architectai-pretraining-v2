# Stage 4.1 — Training Readiness Report

## Scope

This stage hardens corpus reproducibility and prepares deterministic DAPT data and Colab preflight utilities. It does not run DAPT or alter Qwen model weights.

## Correctness and reproducibility

- Production tokenization is pinned to `Qwen/Qwen3-8B` at revision `main`; mock tokenization remains test-only.
- Train/validation assignment and MinHash shingle hashing use SHA-256, removing Python process hash randomization.
- Curated files are staged then atomically replaced after validation.
- Two independent Python-process curation builds were compared. Curated fingerprint, train/validation JSONL SHA-256 values, token counts, and distributions were identical.

## Rebuilt corpus

| Metric | Result |
|---|---:|
| Curated fingerprint | `521acf15e1e2517b7de7a86c0289e61966431fc967829bc72e4cefd0e9794496` |
| Curated documents | 544 |
| Curated tokens | 1,149,730 |
| Train documents / tokens | 524 / 1,117,231 |
| Validation documents / tokens | 20 / 32,499 |
| Train/validation exact ID or normalized-text overlap | 0 |
| Benchmark contamination candidates | 0 |

The rebuilt output is in `data/final/curated/`; its source and manual-review audits are generated alongside it.

## Balance and source coverage

Balancing now measures final retained token shares and records unsatisfied constraints rather than claiming success. The final largest source share is 40.0%, category share 45.4%, and organization share 40.0%. These are quality warnings caused by the available source composition, not hidden passes.

AsciiDoc discovery and title extraction are supported for licensed sources such as Resilience4j. The current raw accepted corpus nevertheless contains zero ADR, DDD, and reliability documents; source audits record this and Stage 2 ingestion must be rerun before those categories can improve.

## DAPT and Colab preparation

- Deterministic causal-LM packing inserts EOS between raw documents, keeps train and validation separate, and masks padding labels as `-100`.
- Packed train output: 546 sequences, 99.96% efficiency; validation: 16 sequences, 99.24% efficiency.
- `configs/dapt.yaml` contains pilot defaults and leaves full-parameter/LoRA/QLoRA selection for actual Colab GPU inspection.
- The Colab package checksum is `8383f5cf17fb9c242a7085cfc5cf624bb65d53c1921f3baf4df75597e4e3b59a` and was verified before model loading.

## Verification

`ruff check src tests scripts`, `mypy src tests`, and 61 pytest tests passed. Pytest emitted one workspace cache permission warning only.

```text
READY_FOR_COLAB_BASELINE=true
READY_FOR_STAGE_5A_SMOKE=true
GO_FOR_FULL_DAPT=false
```

`GO_FOR_FULL_DAPT` remains false until real Colab smoke training validates loss, optimizer updates, checkpoint reload, and resume behavior.

## Logical commit plan

1. `fix(curation): make rebuilds reproducible and auditable` — stable SHA-256 split and MinHash behavior, final-share balancing, atomic curated output, and source/manual audits.
2. `fix(tokenizer): align production defaults with Qwen3` — Qwen3 constants, deterministic mock IDs, EOS support, and exact-count caching.
3. `feat(ingestion): support licensed AsciiDoc sources` — AsciiDoc title handling and regression coverage.
4. `feat(training): add deterministic DAPT preparation and Colab checks` — packing, package verification, preflight, smoke guardrails, configuration, and tests.
5. `docs(training): document Stage 4.1 readiness workflow` — README, artifact policy, and this report.
