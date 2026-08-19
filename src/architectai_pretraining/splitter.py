"""Deterministic train/validation corpus splitter using document ID hashing."""

import hashlib
from dataclasses import dataclass

from architectai_pretraining.models import CorpusDocument


@dataclass
class SplitResult:
    """Output document collections from a train/validation split."""

    train_documents: list[CorpusDocument]
    validation_documents: list[CorpusDocument]


class CorpusSplitter:
    """Splits CorpusDocument objects into train and validation sets deterministically.

    Uses SHA-256 hash of document IDs to assign membership. This guarantees that:
    - Membership is independent of input file ordering or dataset insertion order.
    - No document appears in both splits (zero cross-contamination).
    - Results are 100% reproducible across different environments and runs.
    """

    def __init__(self, train_ratio: float = 0.98, seed: int = 42):
        if not (0.0 < train_ratio < 1.0):
            raise ValueError("train_ratio must be strictly between 0.0 and 1.0")
        self.train_ratio = train_ratio
        self.val_ratio = 1.0 - train_ratio
        self.seed = seed

    def _get_doc_hash_score(self, doc_id: str) -> float:
        """Compute a normalized hash score in [0.0, 1.0) for a document ID."""
        salted_id = f"{self.seed}:{doc_id}"
        hash_hex = hashlib.sha256(salted_id.encode("utf-8")).hexdigest()
        # Convert first 8 bytes of hash to uint32 integer
        val = int(hash_hex[:8], 16)
        return val / 0xFFFFFFFF

    def split(self, documents: list[CorpusDocument]) -> SplitResult:
        if not documents:
            return SplitResult(train_documents=[], validation_documents=[])

        if len(documents) == 1:
            return SplitResult(train_documents=list(documents), validation_documents=[])

        # Assign document to validation if hash score is below val_ratio
        train_docs: list[CorpusDocument] = []
        val_docs: list[CorpusDocument] = []

        # Calculate scores for all documents
        scored_docs = [(self._get_doc_hash_score(doc.id), doc) for doc in documents]

        for score, doc in scored_docs:
            if score < self.val_ratio:
                val_docs.append(doc)
            else:
                train_docs.append(doc)

        # Fallback for small fixture datasets: ensure at least 1 validation doc when len >= 2
        if not val_docs:
            # Pick doc with smallest score for validation split
            scored_docs.sort(key=lambda item: item[0])
            val_docs.append(scored_docs[0][1])
            train_docs = [item[1] for item in scored_docs[1:]]

        elif not train_docs:
            # Pick doc with largest score for train split
            scored_docs.sort(key=lambda item: item[0])
            train_docs.append(scored_docs[-1][1])
            val_docs = [item[1] for item in scored_docs[:-1]]

        return SplitResult(train_documents=train_docs, validation_documents=val_docs)
