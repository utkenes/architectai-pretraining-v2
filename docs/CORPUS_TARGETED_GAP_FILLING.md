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

## Final narrow P1 gap-closing pass

This pass began from `capacity-gapfill-v3` and added no generic architecture or
distributed-systems material. The dominant-source calculation showed that the
remaining independent coverage needed to reach 70% was 4,196 tokens for outbox,
3,779 for bounded-context, and 848 for saga. Outbox was the P0 target.

### Candidate evaluation

| Candidate | Target | License / independence | Decision |
| --- | --- | --- | --- |
| Gruelbox Transaction Outbox | outbox | Apache-2.0; independent Java library maintainer | Selected. Its conceptual guide covers dual writes, rollback, buffering, retries, idempotency, and ordering. |
| Tomorrow One Transactional Outbox | outbox | Apache-2.0; independent Kafka-library maintainer | Selected only after the first source diagnostic yielded 2,591 outbox tokens, leaving roughly 1,605 needed. Its guide adds atomic persistence, ordered relay, at-least-once delivery, consumer deduplication, and failover. |
| AWS Prescriptive Guidance transactional-outbox page | outbox, saga | Technically strong, but documentation licensing was not verified under the repository's release model | Rejected without download. |
| DDD Crew Bounded Context Canvas | bounded-context | CC-BY-4.0 but the same DDD Crew publisher as the existing Context Mapping source | Rejected without download: it would improve source counting but not publisher independence. |

Both selected repositories were sparse snapshots under `ARCHITECT_DATA_DIR` and
are restricted to `README.md` plus their root `LICENSE` evidence. The policy
strips setup, API/reference, configuration, release, and test-oriented sections.

| Source | Snapshot | Accepted contribution | Scope / cap |
| --- | --- | ---: | --- |
| Gruelbox Transaction Outbox Guide | `871430fbb35d930c3b1fcfe7e48e922c5004ff5d` | 2,591 outbox tokens; 5 eligible units | Apache-2.0, README only, 10,000-token cap |
| Tomorrow One Transactional Outbox Guide | `95445248769cb78729697350fede46f9a17abab9` | 1,956 outbox tokens; 1 eligible unit | Apache-2.0, README only, 8,000-token cap |

Manual review confirmed that Gruelbox explains why a local transaction does not
cover external event publication and how rollback/retry behavior changes with
an outbox. Tomorrow One explains the database-plus-event atomic boundary,
ordered relay, at-least-once delivery, and consumer deduplication. Neither
source was manually tagged with a concept; the v3 semantic pipeline assigned
outbox from the retained prose.

### Final capacity audit

Final audit: `data/corpus_v3/capacity-final-gapclose-v3/capacity.json`.

| Measure | Before | After |
| --- | ---: | ---: |
| Configured sources | 33 | 35 |
| Documents discovered | 1,394 | 1,396 |
| Quality-passing units | 3,683 | 3,697 |
| Units after grouping | 2,815 | 2,821 |
| Units after deduplication | 2,804 | 2,810 |
| Eligible tokens | 1,350,486 | 1,355,033 |
| Tokens after source caps | 817,609 | 822,156 |
| Exact / near duplicates removed | 3 / 8 | 3 / 8 |

Messaging/event-driven content rises by 4,547 tokens, from 96,567 to 101,114,
and from 15 to 17 contributing sources. No new source approaches its cap,
category balance is otherwise unchanged, and the candidate concepts remain
`anti-entropy`, `logical-clocks`, and `lease-based-leadership` without automatic
promotion.

| Concept | Tokens before → after | Sources before → after | Documents before → after | Dominant share before → after | Final status |
| --- | --- | --- | --- | --- | --- |
| outbox | 29,089 → 33,636 | 3 → 5 | 13 → 15 | 80.10% → 69.27% | healthy |
| bounded-context | 29,756 → 29,756 | 5 → 5 | 27 → 27 | 78.89% → 78.89% | HIGH_SOURCE_CONCENTRATION |
| domain-event | 29,565 → 29,565 | 4 → 4 | 21 → 21 | 62.82% → 62.82% | healthy |
| saga | 32,885 → 32,885 | 4 → 4 | 20 → 20 | 71.80% → 71.80% | HIGH_SOURCE_CONCENTRATION |

P2 remained diagnostic-only. Gruelbox improved idempotency from 105,922 to
107,421 tokens and reduced its dominant share from 72.52% to 71.51%; caching
rose from 247,144 to 248,345 and fell from 75.26% to 74.89%. Backpressure,
failover, sharding, and concurrency are unchanged. No P2 source was added.

### Preview-readiness decision

**NOT_READY_FOR_20K_PREVIEW.** Outbox is closed and saga is close enough to the
threshold to avoid further collection, but bounded-context remains at 78.89%
from a single dominant source. Its five-source/27-document footprint is useful,
yet this is still a material P1 diversity risk under the configured 70%
diagnostic gate. The next action is a narrowly researched, publisher-independent
bounded-context source; do not generate a preview, freeze, or DAPT artifact
until that decision is resolved.
