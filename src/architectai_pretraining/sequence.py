"""Document length distribution analytics, deterministic chunking, and sequence packing analysis."""

import math
from dataclasses import dataclass, field
from typing import Any

from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.tokenizer import TokenCounter


@dataclass
class LengthPercentiles:
    min_tokens: int
    p10: int
    p25: int
    median: int
    p75: int
    p90: int
    p95: int
    p99: int
    max_tokens: int
    mean_tokens: float


@dataclass
class SequenceLengthEvaluation:
    context_length: int
    natively_fitting_pct: float
    requiring_splitting_pct: float
    estimated_sequence_count: int
    packing_efficiency_pct: float


@dataclass
class DocumentChunk:
    chunk_id: str
    parent_document_id: str
    chunk_index: int
    total_chunks: int
    text: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_length_percentiles(token_counts: list[int]) -> LengthPercentiles:
    if not token_counts:
        return LengthPercentiles(0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)

    s = sorted(token_counts)
    n = len(s)

    def p(pct: float) -> int:
        idx = int(pct * (n - 1))
        return s[min(n - 1, max(0, idx))]

    return LengthPercentiles(
        min_tokens=s[0],
        p10=p(0.10),
        p25=p(0.25),
        median=p(0.50),
        p75=p(0.75),
        p90=p(0.90),
        p95=p(0.95),
        p99=p(0.99),
        max_tokens=s[-1],
        mean_tokens=round(sum(s) / n, 2),
    )


def evaluate_context_length(
    token_counts: list[int], context_length: int
) -> SequenceLengthEvaluation:
    if not token_counts or context_length <= 0:
        return SequenceLengthEvaluation(context_length, 0.0, 0.0, 0, 0.0)

    total_docs = len(token_counts)
    total_tokens = sum(token_counts)

    fitting = sum(1 for c in token_counts if c <= context_length)
    splitting = total_docs - fitting

    natively_fitting_pct = round((fitting / total_docs) * 100.0, 2)
    requiring_splitting_pct = round((splitting / total_docs) * 100.0, 2)

    # Estimate sequence count with chunking
    seq_count = sum(math.ceil(c / context_length) for c in token_counts)

    # Packing efficiency (useful tokens / total allocated capacity)
    total_capacity = seq_count * context_length
    efficiency = (total_tokens / total_capacity) * 100.0 if total_capacity > 0 else 0.0

    return SequenceLengthEvaluation(
        context_length=context_length,
        natively_fitting_pct=natively_fitting_pct,
        requiring_splitting_pct=requiring_splitting_pct,
        estimated_sequence_count=seq_count,
        packing_efficiency_pct=round(efficiency, 2),
    )


class DocumentChunker:
    """Deterministic document chunking strategy preserving section headings."""

    def __init__(self, target_chunk_tokens: int = 2048, overlap_tokens: int = 0) -> None:
        self.target_chunk_tokens = target_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_document(
        self, doc: CorpusDocument, token_counter: TokenCounter
    ) -> list[DocumentChunk]:
        text = doc.text
        doc_tokens = token_counter.count(text)

        if doc_tokens <= self.target_chunk_tokens:
            return [
                DocumentChunk(
                    chunk_id=f"{doc.id}_chunk_001",
                    parent_document_id=doc.id,
                    chunk_index=1,
                    total_chunks=1,
                    text=text,
                    token_count=doc_tokens,
                    metadata=doc.metadata,
                )
            ]

        # Split into paragraphs/sections
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks_text: list[str] = []
        current_paras: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = token_counter.count(para)
            if current_tokens + para_tokens > self.target_chunk_tokens and current_paras:
                chunks_text.append("\n\n".join(current_paras))
                current_paras = [para]
                current_tokens = para_tokens
            else:
                current_paras.append(para)
                current_tokens += para_tokens

        if current_paras:
            chunks_text.append("\n\n".join(current_paras))

        total_chunks = len(chunks_text)
        result: list[DocumentChunk] = []

        for idx, chunk_txt in enumerate(chunks_text, start=1):
            ct_tokens = token_counter.count(chunk_txt)
            result.append(
                DocumentChunk(
                    chunk_id=f"{doc.id}_chunk_{idx:03d}",
                    parent_document_id=doc.id,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    text=chunk_txt,
                    token_count=ct_tokens,
                    metadata=doc.metadata,
                )
            )

        return result
