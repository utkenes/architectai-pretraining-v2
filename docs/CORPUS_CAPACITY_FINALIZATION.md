# Corpus capacity finalization and concept gap audit

Generated with the current v3 code and 30 local configured sources:

```powershell
architectai-pretraining corpus capacity --config configs/corpus_v2.yaml --corpus-root D:\architect-data --output-dir data/corpus_v3/capacity
```

The generated directory is intentionally ignored. All six JSON reports parsed
successfully and their embedded copies reconciled with `capacity.json`.

## Final funnel

| Stage | Count |
| --- | ---: |
| Documents discovered | 1,374 |
| Sections generated | 18,138 |
| Quality-passing units before grouping | 3,615 |
| Units after same-document grouping | 2,767 |
| Units after exact/near deduplication | 2,756 |
| Eligible tokens | 1,326,267 |
| Tokens after source caps | 793,390 |
| Exact duplicates removed | 3 |
| Near duplicates removed | 8 |

The capacity path now applies the same-document grouping used by corpus build,
so its unit/token reports are comparable to preview/freeze preparation.

## Coverage assessment

The 34 canonical concepts have no `NO_COVERAGE`, low-token, low-source, or
low-document status under the configured thresholds. The current risks are
source concentration: `domain-event`, `backpressure`, `outbox`,
`bounded-context`, `saga`, `idempotency`, `failover`, `sharding`,
`concurrency`, and `caching` all exceed the 70% dominant-source threshold.

Priority recommendations:

- P1: add independent explanatory material for `outbox`, `bounded-context`,
  `domain-event`, and `saga`; their volumes are usable but one source supplies
  76–94% of each concept.
- P2: diversify `backpressure`, `idempotency`, `failover`, `sharding`,
  `concurrency`, and `caching`; each has substantial volume and several
  sources, but is still dominated by the Engineering Handbook.
- Candidate review only: `anti-entropy` and `logical-clocks` merit a taxonomy
  decision; `lease-based-leadership` should remain a candidate until supported
  by more than one source.

No candidate was promoted automatically. The primary recommendation is **B —
Targeted Corpus Gap Filling**: add the smallest number of independent sources
that address the P1 diversity gaps, then repeat capacity before generating the
20k preview.
