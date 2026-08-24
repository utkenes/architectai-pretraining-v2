"""Tests for deterministic corpus splitter."""

import pytest

from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.splitter import CorpusSplitter, GroupCorpusSplitter


def test_deterministic_split_order_invariance() -> None:
    splitter = CorpusSplitter(train_ratio=0.5, seed=42)

    docs = [
        CorpusDocument(
            id=f"doc-{i}", source_id="src-1", category="cat-1", text=f"Sample text for doc {i}"
        )
        for i in range(10)
    ]

    res1 = splitter.split(docs)

    # Reverse input order
    reversed_docs = list(reversed(docs))
    res2 = splitter.split(reversed_docs)

    train_ids_1 = {d.id for d in res1.train_documents}
    val_ids_1 = {d.id for d in res1.validation_documents}

    train_ids_2 = {d.id for d in res2.train_documents}
    val_ids_2 = {d.id for d in res2.validation_documents}

    assert train_ids_1 == train_ids_2
    assert val_ids_1 == val_ids_2
    assert len(train_ids_1.intersection(val_ids_1)) == 0


def test_splitter_preserves_all_documents() -> None:
    splitter = CorpusSplitter(train_ratio=0.8, seed=42)
    docs = [
        CorpusDocument(
            id=f"doc-{i}",
            source_id="src-1",
            category="cat-1",
            text=f"Sample content for document number {i}",
        )
        for i in range(20)
    ]

    res = splitter.split(docs)
    total_split_docs = len(res.train_documents) + len(res.validation_documents)
    assert total_split_docs == 20

    all_split_ids = {d.id for d in res.train_documents} | {d.id for d in res.validation_documents}
    assert all_split_ids == {d.id for d in docs}


def test_group_splitter_populates_all_nonzero_splits_without_crossing_provenance_groups() -> None:
    docs = [
        CorpusDocument(
            id=f"doc-{group}-{part}",
            source_id="source",
            category="category",
            text=f"Architecture prose for group {group} part {part}.",
            metadata={"provenance_group_id": f"group-{group}"},
        )
        for group in range(29)
        for part in range(2)
    ]
    splitter = GroupCorpusSplitter(0.90, 0.05, 0.05, seed=7)
    first = splitter.split(docs, require_all_nonempty=True)
    second = splitter.split(list(reversed(docs)), require_all_nonempty=True)

    splits = (first.train_documents, first.validation_documents, first.heldout_documents)
    assert all(split for split in splits)
    memberships: dict[str, set[str]] = {}
    for split_name, split_docs in zip(("train", "validation", "heldout"), splits, strict=True):
        for doc in split_docs:
            memberships.setdefault(doc.metadata["provenance_group_id"], set()).add(split_name)
    assert len(memberships) == 29
    assert all(len(group_splits) == 1 for group_splits in memberships.values())
    assert {
        name: {doc.id for doc in split_docs}
        for name, split_docs in zip(("train", "validation", "heldout"), splits, strict=True)
    } == {
        name: {doc.id for doc in split_docs}
        for name, split_docs in zip(
            ("train", "validation", "heldout"),
            (second.train_documents, second.validation_documents, second.heldout_documents),
            strict=True,
        )
    }


def test_group_splitter_fails_freeze_when_too_few_groups_exist() -> None:
    docs = [
        CorpusDocument(
            id=f"doc-{group}",
            source_id="source",
            category="category",
            text="Architecture prose.",
            metadata={"provenance_group_id": f"group-{group}"},
        )
        for group in range(2)
    ]
    with pytest.raises(ValueError, match="Freeze split integrity requires"):
        GroupCorpusSplitter(0.90, 0.05, 0.05, seed=7).split(docs, require_all_nonempty=True)
