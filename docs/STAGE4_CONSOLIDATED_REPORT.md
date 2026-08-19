# ArchitectAI Stage 4.1–4.2 Consolidated Training Readiness Report

## Verdict

```text
READY_FOR_COLAB=true
READY_TO_RUN_REAL_BASELINE=true
READY_FOR_STAGE_5A_REAL_SMOKE=true
GO_FOR_FULL_DAPT=false
```

This report consolidates the reproducibility, corpus, licensing, benchmark-isolation, packing, packaging, and Colab-preparation results. No model weights were loaded or trained locally.

## Repository and tokenizer

| Field | Value |
|---|---|
| Build git SHA | Recorded automatically from `git rev-parse HEAD` when the immutable v2 package is created |
| Model/tokenizer | `Qwen/Qwen3-8B` |
| Tokenizer revision | `main` |
| Split and MinHash hashing | SHA-256, process-independent |
| Benchmark v1 fingerprint | `9fceffc690ede6c5df433525a710f661a9733b43c8c4a438b7009b7561adaff5` |

## What changed

- Replaced process-random Python hashing in split and MinHash paths with SHA-256.
- Enforced final-retained-corpus concentration analysis and explicit unsatisfied-constraint reporting.
- Rebuilt Stage 2 from current adapters, including Dapr, MADR, and AsciiDoc-capable Resilience4j ingestion.
- Recognized AsciiDoc headings/source blocks in quality and code/prose scoring.
- Added license audit, source audit, deterministic packing, immutable v2 packaging, and verification.
- Added real CUDA-only Stage 5A smoke runner with full-parameter, LoRA, QLoRA, metrics, checkpoint, and resume paths.

## Raw and curated corpus

| Metric | Stage 4.1 | Final Stage 4.2 |
|---|---:|---:|
| Raw documents | 1,300 | 1,864 |
| Raw tokens | 5,978,273 | 6,928,709 |
| Curated documents | 544 | 470 |
| Curated tokens | 1,149,730 | 826,468 |
| Curated fingerprint | `521acf…e9794496` | `db9f8a3b05583e0335eddd190a96bf8909829d4f01965371b56a51a119c76d83` |
| Train documents / tokens | 524 / 1,117,231 | 451 / 804,889 |
| Validation documents / tokens | 20 / 32,499 | 19 / 21,579 |

The final corpus is smaller because the Stage 2 rebuild exposed a broader source set, while quality filtering and final-share balancing discarded low-quality or over-concentrated material. It is more diverse: ADR, reliability, and Dapr content are now represented.

## Final domain distribution

| Category | Documents | Tokens | Share |
|---|---:|---:|---:|
| cloud_architecture | 186 | 330,630 | 40.01% |
| architecture_patterns | 196 | 310,983 | 37.63% |
| messaging | 50 | 139,931 | 16.93% |
| distributed_systems | 12 | 20,514 | 2.48% |
| adr | 22 | 11,217 | 1.36% |
| database_architecture | 3 | 7,075 | 0.86% |
| reliability | 1 | 6,118 | 0.74% |
| domain_driven_design | 0 | 0 | 0.00% |

Largest source: `java_design_patterns` at 30.06%. Largest category: `cloud_architecture` at 40.01%. The main quality warning is zero DDD coverage; no quality gate or licensing requirement was weakened to inflate it.

## Source and license audit

| Source | Enabled | Curated docs | Tokens | Result |
|---|---:|---:|---:|---|
| system_design_primer | yes | 11 | 62,548 | retained |
| java_design_patterns | yes | 185 | 248,435 | retained |
| etcd_docs | yes | 12 | 20,514 | retained |
| cloud_events_spec | yes | 50 | 139,931 | retained |
| dapr_docs | yes | 126 | 129,185 | recovered |
| kubernetes_keps | yes | 60 | 201,445 | retained/balanced |
| clickhouse_docs | yes | 3 | 7,075 | retained |
| resilience4j_docs | yes | 1 | 6,118 | recovered via AsciiDoc scoring |
| madr_docs | yes | 22 | 11,217 | ADR recovered; templates excluded |
| ddd_crew_resources | no | 0 | 0 | `DDD_DATA_INSUFFICIENT`; links-only repository |
| adr_organization_docs | no | 0 | 0 | CC-BY-NC-SA license is incompatible |

All retained records have verified license evidence: `unverified_final_documents = 0`. License distribution is Apache-2.0 45.38%, MIT 31.42%, CC0-1.0 15.63%, and CC-BY-4.0 7.57% by token share.

## Determinism, leakage, and benchmark isolation

- Two independent curation processes produced identical train and validation JSONL SHA-256 values and the same curated fingerprint.
- Train/validation ID overlap: 0.
- Train/validation normalized-text overlap: 0.
- Near-duplicate overlap: 0 after exact and MinHash/LSH curation.
- Benchmark contamination candidates: 0; the frozen 80-scenario benchmark fingerprint is unchanged.

## Packed DAPT data

Sequence length is 2048 and documents are joined only with EOS boundaries. There are no chat templates or SFT fields.

| Split | Sequences | Input/packed tokens | Padding | Dropped | Efficiency | Fingerprint |
|---|---:|---:|---:|---:|---:|---|
| train | 394 | 805,340 | 1,572 | 0 | 99.81% | `f6e706c8…68d6d12a` |
| validation | 11 | 21,598 | 930 | 0 | 95.87% | `11af6e30…efb94b4c` |

The validation efficiency is below the 98% target solely because its final short sequence is padded; no usable tokens were dropped.

## Dataset package and Colab execution

| Field | Value |
|---|---|
| Package | `architectai_dapt_dataset_v2.zip` |
| Package SHA-256 | `ba3eab6cbe6a2dff50eef9e41e8b7f8d5d7527db0fdc6032da87a51c7ea1e69d` |
| Package verification | passed before model load |
| Smoke strategies | explicit `full_parameter`, `lora`, `qlora` |
| T4 recommendation | QLoRA/adapters or a smaller smoke model; full 8B parameter updates are not guaranteed to fit |

See [COLAB_STAGE5.md](COLAB_STAGE5.md) for the single execution sequence: verify package, preflight GPU, run real 4-bit baseline, select a strategy explicitly, run 10–20 smoke steps, checkpoint, resume, then decide GO/NO-GO.

## Validation and remaining warnings

- `ruff check src tests scripts`: passed.
- `mypy src tests`: passed.
- `pytest`: 62 passed; the only warning is an existing workspace pytest-cache permission warning.
- Real baseline and real smoke training are intentionally deferred to Colab.
- DDD coverage remains zero and requires a future explicitly licensed explanatory source; it is a quality warning, not a local correctness blocker.
