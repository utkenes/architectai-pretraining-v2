"""Data models for domain-adaptive pretraining (DAPT) corpus documents."""

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CorpusDocument(BaseModel):
    """Represents a single continuous text document for causal pretraining.

    This model explicitly stores pretraining prose/code text and provenance metadata.
    It does not contain instruction-tuning or dialog fields (e.g. user, assistant).
    """

    id: str = Field(description="Unique identifier for the document")
    source_id: str = Field(description="Identifier of the originating source definition")
    source_url: str | None = Field(default=None, description="URL or location of the raw source")
    license_id: str | None = Field(default=None, description="Legal license identifier, if known")
    category: str = Field(description="Domain architectural category")
    title: str | None = Field(default=None, description="Document title if available")
    text: str = Field(description="Raw or cleaned document text content")
    language: str = Field(default="en", description="Document language code")
    # Optional v2 fields keep frozen records self-describing while preserving
    # compatibility with the original JSONL schema.
    source_name: str | None = None
    source_path: str | None = None
    relative_path: str | None = None
    verified_license_id: str | None = None
    section_title: str | None = None
    token_count: int | None = None
    content_sha256: str | None = None
    quality_score: float | None = None
    architecture_relevance_score: float | None = None
    code_ratio: float | None = None
    source_priority: float | None = None
    corpus_version: str | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary provenance metadata (authors, commit, ingest date, etc.)",
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Ensure document text is not empty or composed solely of whitespace."""
        if not v or not v.strip():
            raise ValueError("Corpus document text cannot be empty or whitespace only.")
        return v

    @field_validator("source_id", "category")
    @classmethod
    def validate_identifier_not_empty(cls, v: str) -> str:
        """Ensure identifiers are non-empty strings."""
        if not v or not v.strip():
            raise ValueError("Source ID and category must be non-empty strings.")
        return v.strip()

    def to_json_str(self) -> str:
        """Serialize the CorpusDocument to a single JSON line without line breaks."""
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False)

    @classmethod
    def from_json_str(cls, line: str) -> "CorpusDocument":
        """Deserialize a CorpusDocument from a JSON string line."""
        data = json.loads(line)
        return cls.model_validate(data)
