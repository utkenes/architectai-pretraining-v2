# Semantic Corpus v3 to DAPT workflow

The only canonical DAPT input is an explicit Semantic Corpus v3 **freeze**.
`data/final/curated` remains a legacy Stage 3 artifact and is never selected by
the v3 DAPT commands.

```text
capacity → 20k preview → manual inspection → freeze → contamination/readiness
→ pack → immutable package → Colab verification → Stage 5A smoke → behavior gate
```

Preview is diagnostic only and is rejected by DAPT. A freeze must contain its
three isolated splits, a v3 manifest, audit ledger, semantic coverage reports,
source diagnostics, and license audit. The heldout split is never packed for
training or validation. Benchmark data is never corpus input.

After an explicit, manually approved freeze:

```powershell
architectai-pretraining dapt pack --corpus-dir data/corpus_v3/freeze
architectai-pretraining dapt readiness --corpus-dir data/corpus_v3/freeze
architectai-pretraining dapt package-data --corpus-dir data/corpus_v3/freeze
architectai-pretraining dapt final-readiness --corpus-dir data/corpus_v3/freeze
```

`readiness` recomputes the corpus and all three split content fingerprints,
checks hashes for each required audit artifact, validates tokenizer identity and
ID/text/provenance split isolation, and proves that packed train/validation
bytes and manifests bind to this exact freeze. Reusing a pack from another
freeze, changing packed JSONL, or changing an audit file is a blocker.

The contamination gate records exact benchmark reuse, n-gram Jaccard overlap,
and benchmark-side containment evidence (including matched n-gram counts). High
containment requires enough benchmark and matching n-grams, so a short generic
phrase cannot trigger the containment rule by itself. Any blocker prevents the
CLI smoke path from loading a model. A passing readiness report is necessary
but not sufficient for full DAPT; the Stage 5A smoke and behavior gate remain
explicit later decisions.
