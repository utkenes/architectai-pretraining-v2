"""Deterministic causal-LM dataset preparation for raw curated documents."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from architectai_pretraining.models import CorpusDocument


class TrainingTokenizer(Protocol):
    """The minimum real-tokenizer API required by DAPT preparation."""

    @property
    def eos_token_id(self) -> int: ...

    def tokenize(self, text: str) -> list[int]: ...


@dataclass
class PackingStatistics:
    sequence_length: int
    input_token_count: int
    packed_token_count: int
    padding_token_count: int
    dropped_token_count: int
    sequence_count: int
    packing_efficiency: float


@dataclass
class PackedDataset:
    sequences: list[dict[str, list[int]]]
    fingerprint: str
    statistics: PackingStatistics

    def write_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for sequence in self.sequences:
                handle.write(json.dumps(sequence, separators=(",", ":")) + "\n")

    def write_manifest(
        self,
        path: str | Path,
        *,
        source_corpus_fingerprint: str | None = None,
        source_split_fingerprint: str | None = None,
        tokenizer_identifier: str | None = None,
        tokenizer_revision: str | None = None,
    ) -> None:
        """Persist packing metadata and, when supplied, its immutable freeze binding."""
        payload: dict[str, object] = {
            "fingerprint": self.fingerprint,
            "statistics": asdict(self.statistics),
        }
        if any(
            value is not None
            for value in (
                source_corpus_fingerprint,
                source_split_fingerprint,
                tokenizer_identifier,
                tokenizer_revision,
            )
        ):
            payload["source_freeze"] = {
                "corpus_fingerprint": source_corpus_fingerprint,
                "split_fingerprint": source_split_fingerprint,
                "tokenizer": {
                    "identifier": tokenizer_identifier,
                    "revision": tokenizer_revision,
                },
            }
        Path(path).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


def compute_packed_fingerprint(sequences: list[dict[str, list[int]]]) -> str:
    """Fingerprint the canonical packed sequence representation."""
    hasher = hashlib.sha256()
    for sequence in sequences:
        hasher.update(json.dumps(sequence, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


def pack_documents(
    documents: list[CorpusDocument],
    tokenizer: TrainingTokenizer,
    sequence_length: int = 2048,
    pad_token_id: int | None = None,
) -> PackedDataset:
    """Tokenize sorted documents, append EOS per document, and pack deterministically.

    The final sequence is padded and its padding labels are ``-100``. No token is
    discarded, and callers must invoke this independently for train and validation.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    eos = tokenizer.eos_token_id
    pad = eos if pad_token_id is None else pad_token_id
    stream: list[int] = []
    for document in sorted(documents, key=lambda item: item.id):
        tokens = tokenizer.tokenize(document.text)
        if tokens:
            stream.extend(tokens)
            stream.append(eos)
    sequences: list[dict[str, list[int]]] = []
    for start in range(0, len(stream), sequence_length):
        chunk = stream[start : start + sequence_length]
        padding = sequence_length - len(chunk)
        input_ids = chunk + [pad] * padding
        labels = chunk + [-100] * padding
        sequences.append(
            {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(chunk) + [0] * padding}
        )
    stats = PackingStatistics(
        sequence_length=sequence_length,
        input_token_count=len(stream),
        packed_token_count=len(stream),
        padding_token_count=sum(sequence_length - sum(seq["attention_mask"]) for seq in sequences),
        dropped_token_count=0,
        sequence_count=len(sequences),
        packing_efficiency=(len(stream) / (len(sequences) * sequence_length)) if sequences else 0.0,
    )
    return PackedDataset(sequences=sequences, fingerprint=compute_packed_fingerprint(sequences), statistics=stats)
