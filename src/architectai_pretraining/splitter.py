"""Deterministic train/validation corpus splitter using document ID hashing."""

import hashlib
from dataclasses import dataclass

from architectai_pretraining.models import CorpusDocument


@dataclass
class SplitResult:
    """Output document collections from a train/validation split."""

    train_documents: list[CorpusDocument]
    validation_documents: list[CorpusDocument]


@dataclass
class GroupSplitResult:
    train_documents: list[CorpusDocument]
    validation_documents: list[CorpusDocument]
    heldout_documents: list[CorpusDocument]


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


class GroupCorpusSplitter:
    """Deterministic three-way split that never separates a provenance group."""

    def __init__(
        self, train_ratio: float = 0.90, validation_ratio: float = 0.05, heldout_ratio: float = 0.05,
        seed: int = 42,
    ) -> None:
        if min(train_ratio, validation_ratio, heldout_ratio) < 0:
            raise ValueError("Split ratios cannot be negative.")
        if abs(train_ratio + validation_ratio + heldout_ratio - 1.0) > 1e-9:
            raise ValueError("Group split ratios must sum to 1.0.")
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.heldout_ratio = heldout_ratio
        self.seed = seed

    def split(
        self, documents: list[CorpusDocument], *, require_all_nonempty: bool = False
    ) -> GroupSplitResult:
        groups: dict[str, list[CorpusDocument]] = {}
        for doc in documents:
            group_id = str(doc.metadata.get("provenance_group_id") or doc.id)
            groups.setdefault(group_id, []).append(doc)
        active_splits = [
            name
            for name, ratio in (
                ("train", self.train_ratio),
                ("validation", self.validation_ratio),
                ("heldout", self.heldout_ratio),
            )
            if ratio > 0
        ]
        if require_all_nonempty and len(groups) < len(active_splits):
            raise ValueError(
                "Freeze split integrity requires at least one provenance group for each non-zero split; "
                f"found {len(groups)} groups for {len(active_splits)} configured splits."
            )
        assignments: dict[str, list[str]] = {name: [] for name in active_splits}
        for group_id in sorted(groups):
            digest = hashlib.sha256(f"{self.seed}:{group_id}".encode()).digest()
            score = int.from_bytes(digest[:8], "big") / 2**64
            if score < self.train_ratio:
                assignments["train"].append(group_id)
            elif score < self.train_ratio + self.validation_ratio:
                assignments["validation"].append(group_id)
            else:
                assignments["heldout"].append(group_id)
        # Hash assignment remains the default. When enough distinct provenance
        # groups exist, deterministically move one whole group to each empty
        # non-zero split rather than silently producing an unusable preview.
        if len(groups) >= len(active_splits):
            for target in active_splits:
                if assignments[target]:
                    continue
                donors = [name for name in active_splits if len(assignments[name]) > 1]
                if not donors:
                    break
                donor = sorted(donors, key=lambda name: (-len(assignments[name]), name))[0]
                move = sorted(
                    assignments[donor],
                    key=lambda group_id: hashlib.sha256(
                        f"{self.seed}:{target}:{group_id}".encode()
                    ).hexdigest(),
                )[0]
                assignments[donor].remove(move)
                assignments[target].append(move)
        if require_all_nonempty and any(not assignments[name] for name in active_splits):
            raise ValueError("Freeze split integrity could not populate every configured non-zero split.")
        train = [doc for group_id in assignments.get("train", []) for doc in groups[group_id]]
        validation = [doc for group_id in assignments.get("validation", []) for doc in groups[group_id]]
        heldout = [doc for group_id in assignments.get("heldout", []) for doc in groups[group_id]]
        return GroupSplitResult(train, validation, heldout)
