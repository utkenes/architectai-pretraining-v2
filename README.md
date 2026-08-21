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

## Architecture corpus v2 (audit-first)

The external architecture corpus is intentionally read-only and is not stored in
this Git repository. Set its root explicitly on Windows, rather than relying on
the project shortcut:

```powershell
$env:ARCHITECT_DATA_DIR = "D:\architect-data"
architectai-pretraining corpus inventory --config configs/corpus_v2.yaml
architectai-pretraining corpus license-audit --config configs/corpus_v2.yaml
architectai-pretraining corpus capacity --config configs/corpus_v2.yaml
```

Run a 20k preview only after the capacity report shows enough eligible total
and category capacity. `preview` uses the pinned `Qwen/Qwen3-8B` tokenizer and
writes provenance-rich continuous-text JSONL plus an audit ledger; it is not a
freeze:

```powershell
architectai-pretraining corpus preview --config configs/corpus_v2.yaml --target-tokens 20000
architectai-pretraining corpus audit --output-dir data/corpus_v2/preview
```

After human review and a passing capacity/category preflight, and only then,
run:

```powershell
architectai-pretraining corpus freeze --config configs/corpus_v2.yaml --target-tokens 1000000
```

The v2 manifest records explicit per-source policies, licensing status,
deduplication, source/category contribution, and a group-safe train/validation/
held-out split. It never reads `data/benchmark/architectai_v1.jsonl` as corpus
input and it never creates SFT/chat records.

The default v2 configuration is an **experimental local corpus**: all 24 local
sources are available for quality-gated curation, including mixed, restrictive,
and unverified sources. Every record records its actual license evidence and a
`release_eligible` flag. Preview writes both `experimental_manifest.json` and
`release_eligible_manifest.json`; the latter shows what remains after removing
release-ineligible records. `corpus capacity` reports whether a 1M freeze could
meet source caps and category tolerance. A freeze aborts before writing output
when those invariants cannot be met.

The seven local expansion sources are NATS, Resilience4j, MADR, Architecture
Decision Guidance Tool, Open Data Hub ADRs, Context Mapping, and Welcome to
DDD. Their directory names include immutable local snapshot hashes; the config
uses the observed full names under `ARCHITECT_DATA_DIR`. If a snapshot is
renamed, update the config after a fresh inventory rather than guessing a path.
Their include/exclude rules select explanatory architecture prose and retain
the normal relevance, code-ratio, quality, exact-dedup, near-dedup, provenance,
and group-safe split gates.

### Corpus semantic linking v3

V3 keeps `category` for old JSONL consumers, but every new training unit has
`schema_version: 3`, a weak `category_hint`, exactly one content-derived
`primary_category`, `related_concepts`, optional auditable `candidate_concepts`,
ordered `section_headings`, and its extraction policy. The canonical concept
vocabulary is deliberately small and deterministic: aliases such as `fault
tolerance`, `fault_tolerance`, and `fault-tolerance` normalize to one label;
unknown terms are reported, never promoted automatically.

Section grouping is limited to adjacent, related sections from the same source
document and respects the token budget. Cross-source linking is report metadata
only: it never joins prose. `corpus capacity` now writes `concept_coverage.json`,
`category_coverage.json`, `source_concept_matrix.json`, `candidate_concepts.json`,
and `source_diagnostics.json` beside `capacity.json`. Coverage accounts for
tokens, sources, documents, units, and dominant-source share, so a high-volume
single source cannot masquerade as healthy diversity.

Recommended workflow: source diagnostics → capacity/concept coverage → targeted
gap analysis → diagnostic preview → manual review → final capacity → explicit
freeze → DAPT. Do not promote candidate concepts or add sources without review.

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
