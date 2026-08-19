"""Tests for JSONL I/O utilities."""

import pytest

from architectai_pretraining.io import iter_jsonl, read_jsonl, stream_write_jsonl, write_jsonl
from architectai_pretraining.models import CorpusDocument


@pytest.fixture
def sample_documents() -> list[CorpusDocument]:
    return [
        CorpusDocument(
            id="doc-1",
            source_id="src-a",
            category="architecture_patterns",
            title="Outbox Pattern",
            text="The transactional outbox pattern guarantees event delivery.",
            metadata={"tag": "events"},
        ),
        CorpusDocument(
            id="doc-2",
            source_id="src-b",
            category="database_architecture",
            title="Unicode Test 🚀",
            text="Special non-ASCII string with math €100 and unicode characters: 𝛌, 𝛍.",
        ),
    ]


def test_write_and_read_jsonl_roundtrip(
    tmp_path: pytest.TempPathFactory, sample_documents: list[CorpusDocument]
) -> None:
    filepath = tmp_path / "test_corpus.jsonl"  # type: ignore[operator]

    count = write_jsonl(sample_documents, filepath)
    assert count == 2
    assert filepath.exists()

    loaded = read_jsonl(filepath)
    assert len(loaded) == 2
    assert loaded[0].id == "doc-1"
    assert loaded[1].title == "Unicode Test 🚀"
    assert "€100" in loaded[1].text


def test_streaming_jsonl_io(
    tmp_path: pytest.TempPathFactory, sample_documents: list[CorpusDocument]
) -> None:
    filepath = tmp_path / "streaming_corpus.jsonl"  # type: ignore[operator]

    written = stream_write_jsonl(iter(sample_documents), filepath)
    assert written == 2

    streamed_docs = list(iter_jsonl(filepath))
    assert len(streamed_docs) == 2
    assert streamed_docs[0] == sample_documents[0]
    assert streamed_docs[1] == sample_documents[1]


def test_iter_jsonl_malformed_line_raises(tmp_path: pytest.TempPathFactory) -> None:
    filepath = tmp_path / "bad_corpus.jsonl"  # type: ignore[operator]
    valid_doc = CorpusDocument(
        id="doc-1",
        source_id="src-1",
        category="cat-1",
        text="Valid text content for document 1.",
    )
    filepath.write_text(f"{valid_doc.to_json_str()}\n{{invalid_json}}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Error parsing line 2"):
        list(iter_jsonl(filepath))
