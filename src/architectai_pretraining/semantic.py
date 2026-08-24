"""Deterministic semantic metadata for the architecture corpus v3.

This module deliberately uses curated phrase rules rather than a network model:
classification is reproducible, inspectable, and source labels remain priors.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.tokenizer import TokenCounter

# Preserve the established v2 category taxonomy.  The configuration values
# remain hints; these are the only legal final values in v3 output.
CANONICAL_CATEGORIES = (
    "distributed_systems",
    "reliability_resilience",
    "software_architecture",
    "domain_driven_design",
    "messaging_event_driven",
    "database_data_architecture",
    "system_design",
    "networking_systems",
    "adr_decision_reasoning",
    "ai_agent_architecture",
)

# The vocabulary is intentionally compact.  New findings are reported as candidates,
# never silently added here.
CANONICAL_CONCEPTS = frozenset(
    {
        "consensus",
        "replication",
        "leader-election",
        "quorum",
        "consistency",
        "eventual-consistency",
        "partitioning",
        "sharding",
        "fault-tolerance",
        "failure-detection",
        "failover",
        "recovery",
        "availability",
        "scalability",
        "caching",
        "load-balancing",
        "messaging",
        "streaming",
        "idempotency",
        "backpressure",
        "transactions",
        "distributed-transactions",
        "saga",
        "outbox",
        "observability",
        "tracing",
        "security",
        "bounded-context",
        "aggregate",
        "domain-event",
        "api-design",
        "storage",
        "indexing",
        "concurrency",
    }
)


def iter_known_concepts() -> tuple[str, ...]:
    """Stable vocabulary order used by coverage reporting."""
    return tuple(sorted(CANONICAL_CONCEPTS))


_ALIASES: dict[str, tuple[str, ...]] = {
    "fault-tolerance": ("fault tolerance", "fault tolerant", "fault_tolerance"),
    "leader-election": ("leader election", "leader_election"),
    "consistency": ("consistency", "strong consistency"),
    "eventual-consistency": ("eventual consistency", "eventual_consistency"),
    "failure-detection": ("failure detection", "failure_detector", "failure detector"),
    "load-balancing": ("load balancing", "load_balancing"),
    "distributed-transactions": ("distributed transactions", "distributed transaction"),
    "bounded-context": ("bounded context", "bounded_context"),
    "domain-event": ("domain event", "domain events", "domain_event"),
    "api-design": ("api design", "api_design", "api architecture"),
    "outbox": ("transactional outbox", "outbox pattern"),
    "saga": ("saga pattern",),
    "quorum": ("quorum",),
    "consensus": ("consensus", "paxos", "raft"),
    "replication": ("replication", "replicated log", "log replication"),
    "partitioning": ("partitioning", "data partition", "partition key"),
    "sharding": (
        "sharding",
        "shard",
    ),
    "availability": ("availability", "high availability"),
    "scalability": ("scalability", "scalable", "scale out"),
    "caching": ("caching", "cache"),
    "messaging": ("messaging", "message broker", "message queue"),
    "streaming": ("streaming", "event stream"),
    "idempotency": ("idempotency", "idempotent"),
    "backpressure": ("backpressure", "back pressure"),
    "transactions": ("transaction", "transactions", "atomicity"),
    "observability": ("observability", "metrics", "monitoring"),
    "tracing": ("tracing", "trace context", "distributed trace"),
    "security": ("security", "authentication", "authorization", "threat model"),
    "aggregate": ("aggregate", "aggregates"),
    "storage": ("storage", "persistent storage"),
    "indexing": ("indexing", "database index", "indexes"),
    "concurrency": ("concurrency", "concurrent", "race condition"),
    "failover": ("failover", "active-passive"),
    "recovery": ("recovery", "disaster recovery"),
}

_CATEGORY_SIGNALS: dict[str, tuple[str, ...]] = {
    "distributed_systems": (
        "consensus",
        "replication",
        "leader-election",
        "quorum",
        "consistency",
        "failure-detection",
    ),
    "reliability_resilience": (
        "fault-tolerance",
        "failover",
        "recovery",
        "availability",
        "backpressure",
    ),
    "messaging_event_driven": ("messaging", "streaming", "idempotency", "outbox", "saga"),
    "domain_driven_design": ("bounded-context", "aggregate", "domain-event"),
    "database_data_architecture": (
        "storage",
        "indexing",
        "sharding",
        "partitioning",
        "transactions",
    ),
    "software_architecture": ("api-design", "security", "observability", "tracing"),
    "system_design": ("scalability", "caching", "load-balancing"),
}

_HINT_CATEGORY_MAP = {
    "architecture_patterns": "software_architecture",
    "cloud_architecture": "software_architecture",
    "adr": "adr_decision_reasoning",
}


def normalize_concept(value: str) -> str | None:
    """Normalize safely; only aliases and canonical vocabulary are returned."""
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    for canonical, aliases in _ALIASES.items():
        forms = {
            re.sub(r"[^a-z0-9]+", "-", item.casefold()).strip("-") for item in (*aliases, canonical)
        }
        if normalized in forms:
            return canonical
    return normalized if normalized in CANONICAL_CONCEPTS else None


def _contains(text: str, phrase: str) -> bool:
    # Separator normalization is intentionally bounded to curated aliases;
    # it never turns arbitrary similar strings into canonical concepts.
    words = [re.escape(word) for word in re.split(r"[-_\s]+", phrase.casefold()) if word]
    return bool(re.search(r"(?<![a-z0-9])" + r"[-_\s]+".join(words) + r"(?![a-z0-9])", text.casefold()))


def discover_concepts(text: str) -> tuple[list[str], list[str]]:
    found = [
        concept
        for concept in CANONICAL_CONCEPTS
        if any(_contains(text, item) for item in (*_ALIASES.get(concept, ()), concept))
    ]
    # Conservative candidate extraction from named architecture phrases. These
    # are audit hints, not labels; one-off generic words are deliberately ignored.
    candidate_aliases = {"logical-clock": "logical-clocks", "logical-clocks": "logical-clocks"}
    candidates = {
        candidate_aliases.get(re.sub(r"\s+", "-", phrase.casefold()), re.sub(r"\s+", "-", phrase.casefold()))
        for phrase in re.findall(
            r"\b(?:anti-entropy|logical clocks?|clock ordering|lease-based leadership)\b",
            text,
            re.I,
        )
        if normalize_concept(phrase) is None
    }
    return sorted(found), sorted(candidates)


def category_coverage_report(docs: Iterable[CorpusDocument]) -> dict[str, dict[str, int]]:
    """Category-level view retaining the same diversity dimensions as concepts."""
    grouped: dict[str, list[CorpusDocument]] = defaultdict(list)
    for doc in docs:
        grouped[doc.primary_category or doc.category].append(doc)
    return {
        category: {
            "tokens": sum(doc.token_count or 0 for doc in items),
            "sources": len({doc.source_id for doc in items}),
            "documents": len({(doc.source_id, doc.relative_path or doc.id) for doc in items}),
            "training_units": len(items),
        }
        for category, items in sorted(grouped.items())
    }


@dataclass(frozen=True)
class Classification:
    primary_category: str | None
    confidence: float
    evidence: list[str]


def classify(primary_text: str, concepts: list[str], category_hint: str | None) -> Classification:
    scores: Counter[str] = Counter()
    evidence: dict[str, list[str]] = defaultdict(list)
    for category, signals in _CATEGORY_SIGNALS.items():
        for concept in concepts:
            if concept in signals:
                scores[category] += 3
                evidence[category].append(concept)
    lowered = primary_text.casefold()
    for category in CANONICAL_CATEGORIES:
        label = category.replace("_", " ").replace("-", " ")
        if label in lowered:
            scores[category] += 1
            evidence[category].append(label)
    hint = _HINT_CATEGORY_MAP.get(category_hint or "", category_hint or "")
    if not scores:
        if hint in CANONICAL_CATEGORIES:
            return Classification(hint, 0.1, ["fallback:category_hint", hint])
        return Classification(None, 0.0, ["unresolved:no_semantic_signal_or_valid_hint"])
    if hint in CANONICAL_CATEGORIES:
        scores[hint] += 0.5
        evidence[hint].append("category_hint")
    winner = sorted(scores, key=lambda category: (-scores[category], category))[0]
    total = sum(scores.values())
    return Classification(winner, round(scores[winner] / total, 3), sorted(set(evidence[winner])))


def annotate_document(doc: CorpusDocument) -> CorpusDocument:
    headings = doc.section_headings or ([doc.section_title] if doc.section_title else [])
    searchable = "\n".join([*headings, doc.title or "", doc.text])
    concepts, candidates = discover_concepts(searchable)
    hint = str(doc.metadata.get("section_category_hint") or doc.category_hint or doc.metadata.get("category_hint") or doc.category)
    result = classify(searchable, concepts, hint)
    unresolved = result.primary_category is None
    return doc.model_copy(
        update={
            "schema_version": 3,
            "category_hint": hint,
            "primary_category": result.primary_category,
            # CorpusDocument keeps a non-empty legacy category field. An
            # unresolved record never reaches selection; its legacy value is
            # retained only so the rejection can be audited.
            "category": result.primary_category or doc.category,
            "related_concepts": concepts,
            "candidate_concepts": candidates,
            "section_headings": headings,
            "extraction_policy": doc.extraction_policy
            or str(doc.metadata.get("extraction_policy", "default")),
            "category_confidence": result.confidence,
            "category_evidence": result.evidence,
            "metadata": {
                **doc.metadata,
                "category_hint": hint,
                "semantic_schema_version": 3,
                "classification_fallback": "fallback:category_hint" in result.evidence,
                "classification_unresolved": unresolved,
            },
        }
    )


def annotate_semantics(doc: CorpusDocument) -> CorpusDocument:
    """Compatibility name for early v3 integrations."""
    return annotate_document(doc)


def related_for_grouping(left: CorpusDocument, right: CorpusDocument) -> bool:
    left_group = left.metadata.get("provenance_group_id")
    right_group = right.metadata.get("provenance_group_id")
    if (
        left.source_id != right.source_id
        or left.relative_path != right.relative_path
        or (left_group is not None and right_group is not None and left_group != right_group)
    ):
        return False
    if set(left.related_concepts) & set(right.related_concepts):
        return True
    if left.primary_category != right.primary_category:
        return False
    ignored = {"and", "the", "with", "that", "this", "from", "into", "uses", "use"}
    left_terms = {
        term
        for term in re.findall(r"[a-z][a-z-]{3,}", left.text.casefold())
        if term not in ignored
    }
    right_terms = {
        term
        for term in re.findall(r"[a-z][a-z-]{3,}", right.text.casefold())
        if term not in ignored
    }
    # This is deliberately a weak same-document fallback, not arbitrary
    # same-category grouping: adjacent prose must share a meaningful term.
    return bool(left_terms & right_terms)


def semantically_related(left: CorpusDocument, right: CorpusDocument) -> bool:
    """Compatibility name for same-document grouping callers."""
    return related_for_grouping(left, right)


def group_adjacent_sections(
    docs: Iterable[CorpusDocument], counter: TokenCounter, max_tokens: int
) -> list[CorpusDocument]:
    """Join only adjacent semantically-linked sections in one source document."""
    ordered = sorted(
        docs,
        key=lambda doc: (
            doc.source_id,
            doc.relative_path or "",
            int(doc.metadata.get("section_index", 0)),
            doc.id,
        ),
    )
    grouped: list[CorpusDocument] = []
    current: CorpusDocument | None = None
    for doc in ordered:
        if current is None:
            current = doc
            continue
        joined = current.text + "\n\n" + doc.text
        contiguous = (
            int(doc.metadata.get("section_index", 0))
            == int(
                current.metadata.get("last_section_index", current.metadata.get("section_index", 0))
            )
            + 1
        )
        if (
            contiguous
            and related_for_grouping(current, doc)
            and counter.count(joined) <= max_tokens
        ):
            headings = [*current.section_headings, *doc.section_headings]
            current_ids = list(current.metadata.get("grouped_section_ids", [current.id]))
            next_ids = list(doc.metadata.get("grouped_section_ids", [doc.id]))
            rescue_metadata = {
                key: value
                for key in ("recall_decision", "rescue_reason")
                for value in (current.metadata.get(key), doc.metadata.get(key))
                if value is not None
            }
            current = annotate_document(
                current.model_copy(
                    update={
                        "id": current.id,
                        "text": joined,
                        "section_headings": headings,
                        "metadata": {
                            **current.metadata,
                            **rescue_metadata,
                            "last_section_index": doc.metadata.get("section_index", 0),
                            "grouped_section_ids": [*current_ids, *next_ids],
                        },
                    }
                )
            )
        else:
            grouped.append(current)
            current = doc
    if current is not None:
        grouped.append(current)
    return grouped


def coverage_report(
    docs: Iterable[CorpusDocument],
    *,
    min_tokens: int = 1_000,
    min_sources: int = 2,
    min_documents: int = 2,
    max_dominant_source_share: float = 0.70,
) -> dict[str, Any]:
    records = list(docs)
    concepts: dict[str, dict[str, Any]] = {}
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for concept in sorted(CANONICAL_CONCEPTS):
        matched = [doc for doc in records if concept in doc.related_concepts]
        tokens = sum(doc.token_count or 0 for doc in matched)
        sources = sorted({doc.source_id for doc in matched})
        documents = sorted({(doc.source_id, doc.relative_path or doc.id) for doc in matched})
        source_tokens = Counter(
            {
                source: sum(doc.token_count or 0 for doc in matched if doc.source_id == source)
                for source in sources
            }
        )
        dominant = round(max(source_tokens.values(), default=0) / max(tokens, 1), 4)
        statuses: list[str] = []
        if not matched:
            statuses.append("NO_COVERAGE")
        else:
            if tokens < min_tokens:
                statuses.append("LOW_TOKENS")
            if len(sources) < min_sources:
                statuses.append("LOW_SOURCE_DIVERSITY")
            if len(documents) < min_documents:
                statuses.append("LOW_DOCUMENT_DIVERSITY")
            if dominant > max_dominant_source_share:
                statuses.append("HIGH_SOURCE_CONCENTRATION")
        concepts[concept] = {
            "concept": concept,
            "tokens": tokens,
            "sources": len(sources),
            "documents": len(documents),
            "training_units": len(matched),
            "dominant_source_share": dominant,
            "dominant_source": source_tokens.most_common(1)[0][0] if source_tokens else None,
            "status": "healthy" if not statuses else statuses,
        }
        for source in sources:
            matrix[concept][source] = source_tokens[source]
    candidates: dict[str, dict[str, Any]] = {}
    for candidate in sorted({item for doc in records for item in doc.candidate_concepts}):
        matched = [doc for doc in records if candidate in doc.candidate_concepts]
        candidates[candidate] = {
            "candidate_concept": candidate,
            "count": len(matched),
            "tokens": sum(doc.token_count or 0 for doc in matched),
            "source_count": len({doc.source_id for doc in matched}),
            "document_count": len({(doc.source_id, doc.relative_path or doc.id) for doc in matched}),
            "sources": sorted({doc.source_id for doc in matched}),
            "documents": sorted(
                {f"{doc.source_id}:{doc.relative_path or doc.id}" for doc in matched}
            ),
            "sample_sections": [doc.section_headings for doc in matched[:3]],
        }
    return {
        "concept_coverage": concepts,
        "category_coverage": category_coverage_report(records),
        "source_concept_matrix": {
            key: dict(sorted(value.items())) for key, value in sorted(matrix.items())
        },
        "candidate_concepts": candidates,
    }
