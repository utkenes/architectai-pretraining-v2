"""End-to-end corpus building pipeline orchestration."""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from architectai_pretraining.cleaner import BaseCleaner, TextCleaner
from architectai_pretraining.dedup import BaseDeduplicator, ExactDeduplicator
from architectai_pretraining.io import stream_write_jsonl
from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.quality import QualityGate, QualityGateConfig
from architectai_pretraining.sources import (
    get_adapter,
    load_source_manifest,
)
from architectai_pretraining.splitter import CorpusSplitter
from architectai_pretraining.stats import CorpusStats, calculate_stats


@dataclass
class PipelineConfig:
    """Configuration for running the pretraining corpus pipeline."""

    manifest_path: str | Path = "configs/sources.yaml"
    raw_dir: str | Path = "data/raw"
    cache_dir: str | Path = field(
        default_factory=lambda: os.getenv(
            "ARCHITECTAI_CACHE_DIR", str(Path(tempfile.gettempdir()) / "architectai_git_cache")
        )
    )
    cleaned_dir: str | Path = "data/cleaned"
    final_dir: str | Path = "data/final"
    train_ratio: float = 0.98
    seed: int = 42
    quality_config: QualityGateConfig | None = None


class CorpusPipeline:
    """Orchestrates ingestion, cleaning, deduplication, quality filtering, and splitting."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        cleaner: BaseCleaner | None = None,
        deduplicator: BaseDeduplicator | None = None,
        quality_gate: QualityGate | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.cleaner = cleaner or TextCleaner()
        self.deduplicator = deduplicator or ExactDeduplicator()
        self.quality_gate = quality_gate or QualityGate(self.config.quality_config)
        self.splitter = CorpusSplitter(train_ratio=self.config.train_ratio, seed=self.config.seed)

    def dry_run(self) -> dict[str, Any]:
        """Perform a dry-run preview listing what would be ingested without downloading."""
        manifest_path = Path(self.config.manifest_path)
        source_configs = load_source_manifest(manifest_path)

        preview_sources: list[dict[str, Any]] = []
        total_candidate_files = 0

        for s_cfg in source_configs:
            entry: dict[str, Any] = {
                "id": s_cfg.id,
                "name": s_cfg.name,
                "category": s_cfg.category,
                "type": s_cfg.type,
                "enabled": s_cfg.enabled,
                "license_id": s_cfg.license_id,
                "url": s_cfg.url,
                "path": s_cfg.path,
                "candidate_files": [],
                "candidate_count": 0,
            }

            if s_cfg.enabled:
                adapter = get_adapter(s_cfg, self.config.cache_dir)
                candidates = adapter.list_candidate_files()
                entry["candidate_files"] = [p.as_posix() for p in candidates[:10]]
                entry["candidate_count"] = len(candidates)
                total_candidate_files += len(candidates)

            preview_sources.append(entry)

        return {
            "total_sources": len(source_configs),
            "enabled_sources": len([s for s in source_configs if s.enabled]),
            "disabled_sources": len([s for s in source_configs if not s.enabled]),
            "total_candidate_files": total_candidate_files,
            "sources": preview_sources,
        }

    def run(self) -> tuple[CorpusStats, Path, Path]:
        """Execute the full pipeline.

        Returns:
            Tuple of (CorpusStats, train_jsonl_path, validation_jsonl_path)
        """
        manifest_path = Path(self.config.manifest_path)
        source_configs = load_source_manifest(manifest_path)
        enabled_sources = [s for s in source_configs if s.enabled]

        source_names: dict[str, str] = {s.id: s.name for s in source_configs}
        per_source_ingested: dict[str, int] = {}
        per_source_rejected: dict[str, int] = {}
        per_source_duplicates: dict[str, int] = {}

        # 1. Raw Ingestion across configured adapters
        raw_documents: list[CorpusDocument] = []
        for s_cfg in enabled_sources:
            adapter = get_adapter(s_cfg, self.config.cache_dir)
            docs = adapter.ingest()
            raw_documents.extend(docs)
            per_source_ingested[s_cfg.id] = len(docs)

        input_count = len(raw_documents)

        # 2. Cleaning
        cleaned_documents = [self.cleaner.clean_document(doc) for doc in raw_documents]

        if self.config.cleaned_dir:
            cleaned_dir = Path(self.config.cleaned_dir)
            cleaned_dir.mkdir(parents=True, exist_ok=True)
            cleaned_path = cleaned_dir / "cleaned_all.jsonl"
            stream_write_jsonl(cleaned_documents, cleaned_path)

        # 3. Exact Deduplication
        dedup_result = self.deduplicator.deduplicate(cleaned_documents)

        # Track duplicates per source
        deduped_ids = {d.id for d in dedup_result.deduplicated_documents}
        for doc in cleaned_documents:
            if doc.id not in deduped_ids:
                s_id = doc.source_id
                per_source_duplicates[s_id] = per_source_duplicates.get(s_id, 0) + 1

        # 4. Quality Gate Filtering
        accepted_docs, rejected_info = self.quality_gate.filter_documents(
            dedup_result.deduplicated_documents
        )
        rejected_count = len(rejected_info)

        # Track quality rejections per source
        for rej in rejected_info:
            s_id = rej[0].source_id
            per_source_rejected[s_id] = per_source_rejected.get(s_id, 0) + 1

        # 5. Deterministic Train/Validation Split
        split_result = self.splitter.split(accepted_docs)

        # 6. JSONL Export
        final_dir = Path(self.config.final_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        train_path = final_dir / "train.jsonl"
        val_path = final_dir / "validation.jsonl"

        stream_write_jsonl(split_result.train_documents, train_path)
        stream_write_jsonl(split_result.validation_documents, val_path)

        # 7. Statistics Report
        stats = calculate_stats(
            train_docs=split_result.train_documents,
            val_docs=split_result.validation_documents,
            input_count=input_count,
            rejected_count=rejected_count,
            duplicates_removed=dedup_result.duplicates_removed,
            per_source_ingested=per_source_ingested,
            per_source_rejected=per_source_rejected,
            per_source_duplicates=per_source_duplicates,
            source_names=source_names,
        )

        return stats, train_path, val_path
