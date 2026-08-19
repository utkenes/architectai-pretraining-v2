# ArchitectAI Domain Pretraining (`architectai-pretraining`)

A production-quality data ingestion and corpus preparation pipeline for **Domain-Adaptive Pretraining (DAPT)** / continued pretraining of open-weight base LLMs (such as Qwen, Llama, or Mistral) on software architecture domain knowledge.

---

## 1. What is ArchitectAI Domain Pretraining?

**ArchitectAI Domain Pretraining** transforms unstructured software architecture knowledge—design patterns, distributed systems theory, domain-driven design principles, reliability engineering, and architectural decision records (ADRs)—into a clean, continuous-text JSONL corpus for **causal language modeling**.

### Concept Breakdown

* **DAPT / Continued Pretraining (This Repository)**: Teaches foundational domain knowledge, architectural vocabulary, and deep conceptual patterns to an open-weight base LLM through unsupervised causal language modeling (`p(token_t | tokens_{<t})`).
* **SFT (Supervised Fine-Tuning)**: Later teaches the model how to act as an architectural assistant, answer questions, follow instructions, and structure reasoning formats (e.g. `user`/`assistant` dialogs).
* **RAG (Retrieval-Augmented Generation)**: Later supplies real-time, project-specific, or current codebase evidence into the model's context window at inference time.

---

## 2. Pipeline Architecture

```text
Architecture Sources (configs/sources.yaml)
                  ↓
          Download / Ingest
                  ↓
           Normalize & Clean (TextCleaner)
                  ↓
        Exact Deduplicate (ExactDeduplicator SHA-256)
                  ↓
          Quality Gate Filter (QualityGate)
                  ↓
     Deterministic Train/Val Split (CorpusSplitter)
                  ↓
          Train / Validation Corpus
 (data/final/train.jsonl & data/final/validation.jsonl)
                  ↓
          Statistics Report (CorpusStats)
```

---

## 3. Repository Structure

```text
architectai-pretraining/
├── configs/
│   └── sources.yaml              # Source manifest configuration
├── data/
│   ├── raw/
│   │   └── manual/               # Local test fixture documents
│   ├── cleaned/                  # Intermediate cleaned JSONL dumps
│   └── final/                    # Processed train/val corpus output
├── src/
│   └── architectai_pretraining/
│       ├── __init__.py
│       ├── __main__.py           # Package entry point
│       ├── cli.py                # Command-line interface
│       ├── models.py             # CorpusDocument Pydantic model
│       ├── io.py                 # JSONL batch and streaming I/O
│       ├── cleaner.py            # Deterministic text cleaner
│       ├── dedup.py              # SHA-256 exact deduplication
│       ├── quality.py            # Quality gate filtering rules
│       ├── splitter.py           # Deterministic SHA-256 ID splitter
│       ├── stats.py              # Corpus metrics & report generator
│       ├── sources.py            # Source manifest and ingestion adapters
│       └── pipeline.py           # End-to-end pipeline orchestrator
├── tests/                        # Comprehensive pytest test suite
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## 4. Installation

Requires **Python >= 3.11**.

```bash
# Clone and enter directory
cd architectai-pretraining

# Install in editable mode with development tools
pip install -e ".[dev]"
# For real Qwen3 tokenization, baseline inference, or Colab preparation:
pip install -e ".[tokenization,training]"
```

---

## 5. How to Ingest Local Documents

Place Markdown (`.md`) or plain text (`.txt`) documents into `data/raw/manual/` (or configure a custom directory in `configs/sources.yaml`).

Example source entry in `configs/sources.yaml`:

```yaml
sources:
  - id: my_local_docs
    name: "My Local Architecture Notes"
    category: architecture_patterns
    enabled: true
    type: local_directory
    path: "data/raw/manual"
    license_id: "CC-BY-4.0"
    url: "file://data/raw/manual"
    language: en
```

---

## 6. How to Build the Corpus

Run the pipeline using the installed CLI or python module:

```bash
architectai-pretraining build
```

or via python module:

```bash
python -m architectai_pretraining build
```

Custom parameters:

```bash
architectai-pretraining build --manifest configs/sources.yaml --train-ratio 0.98 --seed 42
```

Outputs will be saved to:
* `data/final/train.jsonl`
* `data/final/validation.jsonl`

---

## 7. How to Inspect Stats and Validate Output

To print detailed metrics on an existing build:

```bash
architectai-pretraining stats
```

To validate line-by-line schema integrity of generated JSONL splits:

```bash
architectai-pretraining validate
```

---

## 8. Output JSONL Format

Each record in `train.jsonl` and `validation.jsonl` is a serialized `CorpusDocument`:

```json
{
  "id": "a1b2c3d4e5f67890",
  "source_id": "test_fixtures_local",
  "source_url": "file://d:/architectai-pretraining/data/raw/manual/outbox_pattern.md",
  "license_id": "CC-BY-4.0",
  "category": "architecture_patterns",
  "title": "Transactional Outbox Pattern",
  "text": "# Transactional Outbox Pattern\n\nThe Transactional Outbox pattern resolves the dual-write problem...",
  "language": "en",
  "metadata": {
    "file_name": "outbox_pattern.md",
    "relative_path": "outbox_pattern.md",
    "source_name": "Local Test Fixture Architecture Documents (Test Only)",
    "is_test_fixture": true
  }
}
```

Notice:
* `text` contains pure, continuous pretraining text.
* Explicit provenance (`source_id`, `source_url`, `license_id`) is retained.
* No `user`, `assistant`, or `instruction` fields exist.

---

## 9. Licensing and Provenance Expectations

> [!IMPORTANT]
> **Legal Review Requirement**:
> * Provenance metadata (`source_id`, `source_url`, `license_id`) is mandatory.
> * If a source license is unknown, `license_id` remains `null`. Licenses must **never** be inferred or fabricated.
> * All remote or external sources configured in `configs/sources.yaml` must undergo legal review before `enabled: true` is set for model training runs.

---

## 10. Development & Verification

Run static analysis and tests:

```bash
# Check code style & linting
ruff check .

# Type checking
mypy src

# Unit and integration tests
pytest -v
```

---

## 11. What Comes Next

## 12. Stage 4.1 / Colab preparation

Production token counts use the pinned `Qwen/Qwen3-8B` tokenizer at revision `main`; approximate counters are test-only. Curate with the real tokenizer, then create and validate the transfer artifact before any model is loaded:

```bash
architectai-pretraining curate
architectai-pretraining dapt package-data
architectai-pretraining dapt verify-data --manifest data/training/architectai_dapt_dataset_v1.zip
architectai-pretraining dapt preflight
```

The package contains only curated split files and manifests, never raw repository clones, benchmark scenarios, checkpoints, or model weights. `dapt preflight` is deliberately conservative: a T4-class GPU is not treated as suitable for full-parameter Qwen3-8B DAPT. The Stage 5A decision between full-parameter, LoRA, or QLoRA is made only after inspecting the actual Colab hardware. LoRA/QLoRA update adapters, not all base parameters.

`configs/dapt.yaml` defines pilot defaults for raw-document causal language modeling. It does not apply chat templates or create SFT records. Stage 5A smoke execution is intentionally limited to 1–50 steps and must validate finite loss, gradients, checkpoint metadata, and resume behavior before a full DAPT decision.
