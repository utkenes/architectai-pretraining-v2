# Targeted corpus gap filling

## Scope

This sprint adds three independently maintained, prose-first snapshots to improve
the P1 concepts `outbox`, `bounded-context`, `domain-event`, and `saga`. It does
not run preview, freeze, or DAPT, and it does not alter the semantic linker,
quality gates, or deduplication policy.

## Source decisions

| Source | P1 focus | License and snapshot | Scope and cap |
| --- | --- | --- | --- |
| Debezium Outbox Documentation | outbox | CC-BY-3.0 path-scoped documentation; `8aa8f6351911555b95f4e89697ef2c258b781b80` | Two official outbox transformation documents only; 18,000 tokens. The integration/setup document was excluded after review because it was code-heavy. |
| .NET DDD and Domain Events Architecture Guide | bounded-context, domain-event | CC-BY-4.0; `1fd0a4dfd14f5a09bfd071b433409c5998c5ee47` | One Microsoft DDD/CQRS architecture chapter only; 25,000 tokens. Navigation, media, includes, and generic docs are excluded. |
| Eventuate Tram Sagas Guide | saga | Apache-2.0; `e8d57a1c62b052303ab1acee1ddcd4baebdb0562` | Root `README.adoc` only; 24,000 tokens. Modules, tests, and implementation code are excluded. |

The initial EventDriven.Sagas candidate was rejected rather than retained: its
single guide produced only 268 quality-passing tokens. The replacement produces
1,818 eligible tokens from one coherent prose unit and directly covers local
transactions, orchestration, and compensating actions.

Each source is configured with an explicit snapshot SHA, restrictive includes,
content type, license evidence, approval/release status, and a source cap.
External source worktrees live under `ARCHITECT_DATA_DIR`; generated capacity
reports remain under ignored `data/corpus_v3/` paths.

## Focused review

The full capacity pass confirms the source-local funnel below.

| Source | Raw documents | Passing units before grouping | Eligible units | Eligible tokens |
| --- | ---: | ---: | ---: | ---: |
| Debezium | 2 | 21 | 13 | 4,385 |
| .NET DDD | 17 | 46 | 34 | 18,016 |
| Eventuate | 1 | 1 | 1 | 1,818 |

Manual samples were reviewed in the selected upstream files:

- Debezium describes capturing changes in an outbox table and applying the
  outbox event-router transformation.
- The .NET guide distinguishes in-process domain events from asynchronous
  integration events across bounded contexts.
- The Eventuate guide describes sequential local transactions and reverse-order
  compensations when a saga step fails.

## Capacity comparison

Baseline: `data/corpus_v3/capacity/capacity.json`. Targeted run:
`data/corpus_v3/capacity-gapfill-v3/capacity.json`.

| P1 concept | Tokens before | Tokens after | Delta | Sources before → after | Dominant share before → after | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| outbox | 24,704 | 29,089 | +4,385 | 2 → 3 | 94.31% → 80.10% | Improved; still concentrated |
| bounded-context | 29,221 | 29,756 | +535 | 4 → 5 | 80.33% → 78.89% | Improved; still concentrated |
| domain-event | 23,882 | 29,565 | +5,683 | 3 → 4 | 77.77% → 62.82% | Healthy |
| saga | 31,067 | 32,885 | +1,818 | 3 → 4 | 76.01% → 71.80% | Improved; still concentrated |

Overall eligible capacity rises from 1,326,267 to 1,350,486 tokens; post-source-
cap capacity rises from 793,390 to 817,609 tokens. Both changes are +24,219
tokens, and all added eligible content is release eligible. The run discovered
1,394 documents and retained 2,804 units after deduplication (baseline: 1,374
and 2,756); exact and near-duplicate removals remain 3 and 8.

## Next decision

Choose **A: add one more targeted source**. The highest-value follow-up is an
independent bounded-context or outbox source, since those remain above the 70%
dominant-source threshold. No semantic or quality-gate bug was found, so a code
change would not be justified before further source research.
