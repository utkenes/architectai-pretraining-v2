"""Tests for deterministic corpus splitter."""

from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.splitter import CorpusSplitter


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
