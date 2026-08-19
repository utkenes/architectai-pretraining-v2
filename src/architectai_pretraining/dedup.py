"""Deduplication module for exact and near-duplicate document removal."""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

from architectai_pretraining.models import CorpusDocument


@dataclass
class DedupResult:
    """Statistics and output documents from a deduplication pass."""

    input_documents: int
    duplicates_removed: int
    remaining_documents: int
    deduplicated_documents: list[CorpusDocument]


class BaseDeduplicator(ABC):
    """Abstract base class for document deduplication algorithms."""

    @abstractmethod
    def deduplicate(self, documents: list[CorpusDocument]) -> DedupResult:
        """Deduplicate a list of CorpusDocument objects."""
        pass


class ExactDeduplicator(BaseDeduplicator):
    """Exact deduplication based on cryptographic SHA-256 text content hashing.

    Identical normalized document texts map to the same hash.
    The first encountered document is preserved, while subsequent duplicates are filtered out.
    """

    def __init__(self, normalize_whitespace: bool = True):
        self.normalize_whitespace = normalize_whitespace

    def _compute_hash(self, text: str) -> str:
        content = text.strip()
        if self.normalize_whitespace:
            # Collapse internal whitespace runs for robust exact matching
            content = " ".join(content.split())
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def deduplicate(self, documents: list[CorpusDocument]) -> DedupResult:
        seen_hashes: set[str] = set()
        unique_docs: list[CorpusDocument] = []
        input_count = len(documents)

        for doc in documents:
            doc_hash = self._compute_hash(doc.text)
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                unique_docs.append(doc)

        duplicates_removed = input_count - len(unique_docs)

        return DedupResult(
            input_documents=input_count,
            duplicates_removed=duplicates_removed,
            remaining_documents=len(unique_docs),
            deduplicated_documents=unique_docs,
        )
