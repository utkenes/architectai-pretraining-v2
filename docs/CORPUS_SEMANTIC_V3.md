# Corpus semantic linking v3

The corpus pipeline is audit-first and does not start DAPT, freeze a corpus, or
fetch sources as part of semantic analysis. Its v3 flow is:

`source → extraction → heading sectionization → quality gates → primary category → concepts → same-document grouping → dedup → coverage/balance → preview/freeze`

Each accepted v3 training unit has `schema_version: 3`, one
`primary_category`, zero or more `related_concepts`, and zero or more
`candidate_concepts`. The legacy `category` field mirrors `primary_category`
for older consumers. `category_hint` (or legacy source `category`) is a weak,
auditable prior only; evidence from headings and prose can override it.

The canonical concept list is deliberately small and deterministic. Aliases
normalize punctuation, spaces, and underscores, but unknown discoveries are
reported in `candidate_concepts.json`; a capacity run never changes the
taxonomy. Classification confidence and evidence are preserved with the unit.

Section grouping can only join adjacent sections from the same source document,
in their existing order, when the categories/concepts are continuous and the
combined unit is within the token budget. Cross-source links exist only in the
source-concept matrix, never as concatenated training prose.

Run the audit workflow with the external corpus mounted read-only:

```powershell
$env:ARCHITECT_DATA_DIR = "D:\architect-data"
architectai-pretraining corpus inventory --config configs/corpus_v2.yaml --output-dir data/corpus_v3/inventory
architectai-pretraining corpus capacity --config configs/corpus_v2.yaml --output-dir data/corpus_v3/capacity
```

Capacity writes `source_diagnostics.json`, `category_coverage.json`,
`concept_coverage.json`, `candidate_concepts.json`, and
`source_concept_matrix.json`. Concept status distinguishes no coverage, low
tokens, low source/document diversity, and high source concentration. Address
those findings before generating a manually approved preview. The intended
decision sequence is diagnostics → capacity → gap analysis → preview → manual
review → final capacity → freeze → DAPT.
