# Controlled contextual recall

This pass recovers only sections that were structurally weakened by
sectionization. It does not add sources, alter license decisions, change
category targets, relax source caps, or run a corpus build.

Scoring uses an ephemeral view made from document title, section title, section
headings, and the unchanged section text. The view is never written to a DAPT
JSONL record: `CorpusDocument.text` remains the extracted training prose.

The primary path remains strict: a section must satisfy the configured
relevance and quality thresholds and every hard gate. `CodeProseAnalyzer` is
the hard code-dominance authority; generated navigation, project metadata,
installation-only pages, configured policy exclusions, unresolved semantics,
navigation/link dumps, duplicates, and malformed input remain exclusions. Link
ratio remains an audit metric and a soft relevance penalty, not a standalone
hard rejection for substantive technical prose.

Only an `approved` source can enter the named borderline window in
`configs/corpus_v2.yaml` (relevance `>= 0.28`, quality `>= 0.33`). A borderline
unit must then be immediately adjacent to a semantically related unit from the
same source, relative path, and provenance group. Their natural joined text
must fit `max_section_tokens`, and the joined candidate must satisfy the normal
acceptance thresholds when rescored. No cross-document, cross-source, or
unrelated-category rescue exists.

Capacity reports now distinguish source documents from sections and include
normal, borderline, rescued, rejected, and token metrics. They also include
fixed score buckets and a seeded bounded audit sample. `write_capacity` writes
that sample to `rejection_samples.jsonl`; it includes full text only for the
small deterministic sample, not the normal manifest.

Review a post-change capacity report and its accepted/rejected samples before
running a preview or freeze. A code change alone is not evidence that the real
corpus is ready for freeze.
