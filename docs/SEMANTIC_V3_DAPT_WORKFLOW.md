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

`readiness` validates manifest identity, tokenizer identity, ID/text/provenance
split isolation, license metadata, and the benchmark n-gram contamination gate.
Any blocker prevents the CLI smoke path from loading a model. A passing
readiness report is necessary but not sufficient for full DAPT; the Stage 5A
smoke and behavior gate remain explicit later decisions.
