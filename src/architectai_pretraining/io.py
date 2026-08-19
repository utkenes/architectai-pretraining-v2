"""Streaming and batch JSONL I/O utilities for CorpusDocument collections."""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from architectai_pretraining.models import CorpusDocument


def stream_write_jsonl(documents: Iterable[CorpusDocument], filepath: str | Path) -> int:
    """Stream documents to a JSONL file.

    Args:
        documents: Iterable of CorpusDocument objects.
        filepath: Destination file path.

    Returns:
        The total number of documents written.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.to_json_str() + "\n")
            count += 1

    return count


def write_jsonl(documents: list[CorpusDocument], filepath: str | Path) -> int:
    """Write a list of documents to a JSONL file."""
    return stream_write_jsonl(documents, filepath)


def iter_jsonl(filepath: str | Path) -> Iterator[CorpusDocument]:
    """Stream documents from a JSONL file line-by-line.

    Args:
        filepath: Path to the JSONL file.

    Yields:
        CorpusDocument objects.

    Raises:
        ValueError: If a line cannot be parsed as a valid CorpusDocument.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                yield CorpusDocument.from_json_str(line_str)
            except (json.JSONDecodeError, Exception) as e:
                raise ValueError(f"Error parsing line {line_idx} in {path}: {e}") from e


def read_jsonl(filepath: str | Path) -> list[CorpusDocument]:
    """Read all documents from a JSONL file into a list."""
    return list(iter_jsonl(filepath))


def write_dict_jsonl(records: Iterable[dict[str, Any]], filepath: str | Path) -> int:
    """Write an iterable of dictionaries to a JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    return count


def iter_dict_jsonl(filepath: str | Path) -> Iterator[dict[str, Any]]:
    """Stream dictionaries from a JSONL file line-by-line."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                if isinstance(data, dict):
                    yield data
            except Exception as e:
                raise ValueError(f"Error parsing line {line_idx} in {path}: {e}") from e

