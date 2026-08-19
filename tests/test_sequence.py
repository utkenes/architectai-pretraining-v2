"""Unit tests for sequence analytics, length percentiles, and chunking."""

from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.sequence import (
    DocumentChunker,
    compute_length_percentiles,
    evaluate_context_length,
)
from architectai_pretraining.tokenizer import MockTokenCounter


def test_compute_length_percentiles() -> None:
    counts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    p = compute_length_percentiles(counts)
    assert p.min_tokens == 100
    assert p.max_tokens == 1000
    assert p.median >= 400
    assert p.mean_tokens == 550.0


def test_evaluate_context_length() -> None:
    counts = [500, 1000, 1500, 2500, 3000]
    eval_res = evaluate_context_length(counts, 2048)
    assert eval_res.context_length == 2048
    assert eval_res.natively_fitting_pct == 60.0
    assert eval_res.requiring_splitting_pct == 40.0
    assert eval_res.estimated_sequence_count == 7


def test_document_chunker() -> None:
    counter = MockTokenCounter()
    doc = CorpusDocument(
        id="long_doc",
        title="Long Architecture Document",
        text="Section 1\n\n" + ("word " * 1000) + "\n\nSection 2\n\n" + ("word " * 1000),
        source_id="src1",
        category="architecture_patterns",
        license_id="MIT",
    )

    chunker = DocumentChunker(target_chunk_tokens=500)
    chunks = chunker.chunk_document(doc, counter)

    assert len(chunks) >= 2
    assert chunks[0].parent_document_id == "long_doc"
    assert chunks[0].chunk_index == 1


