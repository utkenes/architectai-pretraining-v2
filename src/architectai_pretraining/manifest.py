"""Curation manifest and stable corpus fingerprint generator."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from architectai_pretraining.models import CorpusDocument


def compute_corpus_fingerprint(
    docs: Sequence[CorpusDocument], config_hash: str = ""
) -> str:
    """Generate deterministic SHA-256 fingerprint over ordered document IDs and content hashes."""
    hasher = hashlib.sha256()
    hasher.update(config_hash.encode("utf-8"))

    sorted_docs = sorted(docs, key=lambda d: d.id)
    for doc in sorted_docs:
        hasher.update(doc.id.encode("utf-8"))
        hasher.update(doc.text.encode("utf-8"))
        v_lic = str(doc.metadata.get("verified_license_id") or doc.license_id or "")
        hasher.update(v_lic.encode("utf-8"))

    return hasher.hexdigest()


@dataclass
class CurationManifest:
    pipeline_version: str
    build_timestamp: str
    tokenizer_identifier: str
    tokenizer_revision: str
    input_corpus_fingerprint: str
    output_corpus_fingerprint: str
    curation_config_hash: str
    input_documents_count: int
    curated_documents_count: int
    train_documents_count: int
    validation_documents_count: int
    holdout_documents_count: int
    total_input_tokens: int
    total_curated_tokens: int
    train_tokens: int
    validation_tokens: int
    holdout_tokens: int
    quality_rejects_count: int
    relevance_rejects_count: int
    exact_duplicates_count: int
    near_duplicates_count: int
    balanced_out_count: int
    fixture_excluded_count: int
    source_distribution: dict[str, dict[str, Any]]
    category_distribution: dict[str, dict[str, Any]]
    license_distribution: dict[str, dict[str, Any]]
    quality_bucket_distribution: dict[str, int]
    minhash_lsh_params: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
