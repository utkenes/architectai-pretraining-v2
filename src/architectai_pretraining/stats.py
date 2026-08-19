"""Corpus statistics calculation and per-source reporting."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from architectai_pretraining.models import CorpusDocument


class TokenCounter(Protocol):
    """Protocol for optional future tokenizer integration."""

    def count_tokens(self, text: str) -> int: ...


@dataclass
class SourceStats:
    """Per-source statistics for corpus ingestion and processing."""

    source_id: str
    source_name: str
    documents_ingested: int = 0
    documents_rejected: int = 0
    characters: int = 0
    words: int = 0
    duplicates_removed: int = 0


@dataclass
class CorpusStats:
    """Summary statistics for a pretraining corpus build."""

    input_documents: int = 0
    accepted_documents: int = 0
    rejected_documents: int = 0
    duplicates_removed: int = 0
    train_documents: int = 0
    validation_documents: int = 0
    total_characters: int = 0
    total_words: int = 0
    total_tokens: int | None = None
    avg_char_length: float = 0.0
    avg_word_count: float = 0.0
    documents_by_category: dict[str, int] = field(default_factory=dict)
    documents_by_source: dict[str, int] = field(default_factory=dict)
    source_stats: dict[str, SourceStats] = field(default_factory=dict)

    def to_formatted_report(self) -> str:
        """Format the metrics into a human-readable text summary."""
        lines = [
            "==================================================",
            "        ArchitectAI Corpus Pipeline Report        ",
            "==================================================",
            f"Input Documents:                  {self.input_documents}",
            f"Accepted Documents:               {self.accepted_documents}",
            f"Rejected Documents (Quality):     {self.rejected_documents}",
            f"Exact Duplicates Removed:         {self.duplicates_removed}",
            "--------------------------------------------------",
            f"Train Split Documents:            {self.train_documents}",
            f"Validation Split Documents:       {self.validation_documents}",
            "--------------------------------------------------",
            f"Total Characters:                 {self.total_characters:,}",
            f"Total Words:                      {self.total_words:,}",
        ]

        if self.total_tokens is not None:
            lines.append(f"Total Tokens (Tokenizer):         {self.total_tokens:,}")

        lines.extend(
            [
                f"Average Document Length (chars):  {self.avg_char_length:.1f}",
                f"Average Document Length (words):  {self.avg_word_count:.1f}",
                "--------------------------------------------------",
                "Documents by Category:",
            ]
        )

        for cat, count in sorted(self.documents_by_category.items()):
            lines.append(f"  - {cat}: {count}")

        if self.source_stats:
            lines.extend(
                [
                    "--------------------------------------------------",
                    "Per-Source Statistics:",
                ]
            )
            for s_id, s_stat in sorted(self.source_stats.items()):
                lines.append(
                    f"  - {s_stat.source_name} ({s_id}):\n"
                    f"      Documents Ingested: {s_stat.documents_ingested} | "
                    f"Rejected: {s_stat.documents_rejected} | "
                    f"Duplicates Removed: {s_stat.duplicates_removed}\n"
                    f"      Characters: {s_stat.characters:,} | Words: {s_stat.words:,}"
                )

        lines.append("==================================================")
        return "\n".join(lines)


def calculate_stats(
    train_docs: list[CorpusDocument],
    val_docs: list[CorpusDocument],
    input_count: int = 0,
    rejected_count: int = 0,
    duplicates_removed: int = 0,
    per_source_ingested: dict[str, int] | None = None,
    per_source_rejected: dict[str, int] | None = None,
    per_source_duplicates: dict[str, int] | None = None,
    source_names: dict[str, str] | None = None,
    token_counter: TokenCounter | None = None,
) -> CorpusStats:
    """Calculate statistics across train and validation sets with per-source metrics."""
    all_docs = train_docs + val_docs
    total_docs = len(all_docs)

    total_chars = sum(len(d.text) for d in all_docs)
    total_words = sum(len(d.text.split()) for d in all_docs)

    total_tokens = None
    if token_counter is not None and all_docs:
        total_tokens = sum(token_counter.count_tokens(d.text) for d in all_docs)

    avg_chars = (total_chars / total_docs) if total_docs > 0 else 0.0
    avg_words = (total_words / total_docs) if total_docs > 0 else 0.0

    by_category = dict(Counter(d.category for d in all_docs))
    by_source = dict(Counter(d.source_id for d in all_docs))

    # Compute detailed per-source metrics
    source_stats_map: dict[str, SourceStats] = {}
    known_sources = set(by_source.keys())
    if per_source_ingested:
        known_sources.update(per_source_ingested.keys())

    names_map = source_names or {}

    for s_id in sorted(known_sources):
        docs_for_source = [d for d in all_docs if d.source_id == s_id]
        s_chars = sum(len(d.text) for d in docs_for_source)
        s_words = sum(len(d.text.split()) for d in docs_for_source)
        ingested = (
            per_source_ingested.get(s_id, len(docs_for_source))
            if per_source_ingested
            else len(docs_for_source)
        )
        rejected = per_source_rejected.get(s_id, 0) if per_source_rejected else 0
        dups = per_source_duplicates.get(s_id, 0) if per_source_duplicates else 0

        source_stats_map[s_id] = SourceStats(
            source_id=s_id,
            source_name=names_map.get(s_id, s_id),
            documents_ingested=ingested,
            documents_rejected=rejected,
            characters=s_chars,
            words=s_words,
            duplicates_removed=dups,
        )

    return CorpusStats(
        input_documents=input_count or total_docs,
        accepted_documents=total_docs,
        rejected_documents=rejected_count,
        duplicates_removed=duplicates_removed,
        train_documents=len(train_docs),
        validation_documents=len(val_docs),
        total_characters=total_chars,
        total_words=total_words,
        total_tokens=total_tokens,
        avg_char_length=avg_chars,
        avg_word_count=avg_words,
        documents_by_category=by_category,
        documents_by_source=by_source,
        source_stats=source_stats_map,
    )
