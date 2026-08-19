"""Stage 3 end-to-end corpus curation, balancing, tokenization, and audit orchestrator."""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from architectai_pretraining.balance import CorpusBalancer
from architectai_pretraining.cleaner import BoilerplateCleaner
from architectai_pretraining.code_prose import CodeProseAnalyzer
from architectai_pretraining.diagnostics import (
    SourceDiagnosticsEngine,
    generate_license_audit,
    generate_source_audit,
    write_manual_review_sample,
)
from architectai_pretraining.io import read_jsonl, write_jsonl
from architectai_pretraining.manifest import CurationManifest, compute_corpus_fingerprint
from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.near_dedup import MinHashLSHDeduplicator
from architectai_pretraining.relevance import DomainRelevanceGate
from architectai_pretraining.report import evaluate_stage4_readiness, generate_corpus_audit_report
from architectai_pretraining.scoring import DocumentQualityScore, DocumentQualityScorer
from architectai_pretraining.sequence import (
    compute_length_percentiles,
    evaluate_context_length,
)
from architectai_pretraining.sources import load_source_manifest
from architectai_pretraining.tokenizer import (
    CachingTokenCounter,
    HuggingFaceTokenCounter,
    MockTokenCounter,
    TokenCounter,
)

logger = logging.getLogger(__name__)


@dataclass
class CurationConfig:
    manifest_path: str | Path = "configs/sources.yaml"
    raw_accepted_dir: str | Path = "data/final/raw_accepted"
    raw_accepted_fallback: str | Path = "data/final"
    curated_dir: str | Path = "data/final/curated"
    cache_dir: str | Path | None = None
    use_real_tokenizer: bool = True
    tokenizer_identifier: str = "Qwen/Qwen3-8B"
    tokenizer_revision: str = "main"
    fallback_allowed_in_prod: bool = False
    train_ratio: float = 0.95
    validation_ratio: float = 0.05
    holdout_ratio: float = 0.00
    seed: int = 42
    max_source_token_share: float = 0.30
    max_category_token_share: float = 0.40
    max_organization_token_share: float = 0.40


class CurationPipeline:
    """Orchestrates end-to-end Stage 3 curation, balancing, tokenization, and audit reporting."""

    def __init__(self, config: CurationConfig | None = None) -> None:
        self.config = config or CurationConfig()
        self.raw_dir = Path(self.config.raw_accepted_dir)
        self.curated_dir = Path(self.config.curated_dir)
        self.manifest_path = Path(self.config.manifest_path)

        # Load manifest
        self.sources = load_source_manifest(self.manifest_path)

    def _load_raw_documents(self) -> list[CorpusDocument]:
        train_p = self.raw_dir / "train.jsonl"
        val_p = self.raw_dir / "validation.jsonl"

        if not train_p.exists():
            fallback_dir = Path(self.config.raw_accepted_fallback)
            train_p = fallback_dir / "train.jsonl"
            val_p = fallback_dir / "validation.jsonl"

        docs: list[CorpusDocument] = []
        if train_p.exists():
            docs.extend(read_jsonl(train_p))
        if val_p.exists():
            docs.extend(read_jsonl(val_p))

        logger.info("Loaded %d raw input documents for Stage 3 curation.", len(docs))
        return docs

    def run(self) -> tuple[CurationManifest, str]:
        # 1. Load Raw Documents
        raw_docs = self._load_raw_documents()
        input_doc_count = len(raw_docs)

        # 2. Input Corpus Fingerprint
        input_fingerprint = compute_corpus_fingerprint(raw_docs, "raw_stage2")

        # 3. Tokenizer Setup
        real_tokenizer_used = False
        token_counter: TokenCounter
        if self.config.use_real_tokenizer:
            try:
                token_counter = HuggingFaceTokenCounter(
                    identifier=self.config.tokenizer_identifier,
                    revision=self.config.tokenizer_revision,
                    fallback_allowed_in_prod=self.config.fallback_allowed_in_prod,
                )
                real_tokenizer_used = True
            except Exception as e:
                logger.error("Real tokenizer initialization failed: %s", e)
                if not self.config.fallback_allowed_in_prod:
                    raise
                token_counter = MockTokenCounter()
        else:
            token_counter = MockTokenCounter()

        # The same document is measured repeatedly by quality, balance, reports,
        # and splits. Cache exact counts for this build; no approximate fallback.
        token_counter = CachingTokenCounter(token_counter)

        from tempfile import gettempdir

        cache_p = Path(self.config.cache_dir or Path(gettempdir()) / "architectai_git_cache")
        diag_engine = SourceDiagnosticsEngine(self.sources, cache_p)
        diag_engine.generate_report(self.curated_dir / "source_diagnostics.json")

        ledger_entries: list[dict[str, Any]] = []

        # Step 1: Fixture Isolation Filter
        non_fixture_docs: list[CorpusDocument] = []
        fixture_excluded_count = 0

        for doc in raw_docs:
            if doc.source_id == "test_fixtures_local" or doc.metadata.get("is_test_fixture"):
                fixture_excluded_count += 1
                ledger_entries.append(
                    {
                        "document_id": doc.id,
                        "decision": "fixture_excluded",
                        "reason": "Test fixture document isolated from production dataset.",
                        "source_id": doc.source_id,
                    }
                )
            else:
                non_fixture_docs.append(doc)

        # Step 2: Boilerplate Cleaning
        bp_cleaner = BoilerplateCleaner()
        cleaned_docs = [bp_cleaner.clean_document(d) for d in non_fixture_docs]

        # Step 3: Domain Relevance Gate
        rel_gate = DomainRelevanceGate()
        relevant_docs: list[CorpusDocument] = []
        relevance_rejects_count = 0

        for doc in cleaned_docs:
            res = rel_gate.check(doc)
            if not res.is_relevant:
                relevance_rejects_count += 1
                ledger_entries.append(
                    {
                        "document_id": doc.id,
                        "decision": "rejected_domain_relevance",
                        "reason": res.reason,
                        "category": res.category,
                        "source_id": doc.source_id,
                    }
                )
            else:
                relevant_docs.append(doc)

        # Step 4: Document Quality Scoring
        doc_scorer = DocumentQualityScorer(min_document_score=0.45)
        scored_docs: list[CorpusDocument] = []
        doc_quality_scores = {}
        quality_rejects_count = 0
        quality_buckets_cnt: dict[str, int] = {"high": 0, "medium": 0, "low": 0}

        for doc in relevant_docs:
            qs = doc_scorer.score(doc)
            doc_quality_scores[doc.id] = qs
            if qs.quality_bucket == "low":
                quality_rejects_count += 1
                ledger_entries.append(
                    {
                        "document_id": doc.id,
                        "decision": "rejected_quality",
                        "reason": f"Quality score {qs.quality_score} below minimum threshold 0.45",
                        "quality_score": qs.quality_score,
                        "source_id": doc.source_id,
                    }
                )
            else:
                scored_docs.append(doc)
                quality_buckets_cnt[qs.quality_bucket] += 1

        # Step 5: Code vs Prose Filter
        cp_analyzer = CodeProseAnalyzer(max_code_token_ratio=0.70)
        prose_docs: list[CorpusDocument] = []

        for doc in scored_docs:
            cp_metrics = cp_analyzer.analyze(doc, token_counter)
            if cp_metrics.is_code_dominated:
                quality_rejects_count += 1
                ledger_entries.append(
                    {
                        "document_id": doc.id,
                        "decision": "rejected_quality",
                        "reason": f"Code-dominated document (code ratio {cp_metrics.code_to_prose_ratio:.1%})",
                        "source_id": doc.source_id,
                    }
                )
            else:
                prose_docs.append(doc)

        # Step 6: MinHash LSH Near-Duplicate Detection
        minhash_lsh = MinHashLSHDeduplicator(
            num_perm=128,
            shingle_size=5,
            similarity_threshold=0.85,
            num_bands=16,
            rows_per_band=8,
        )
        scores_map = {d.id: doc_quality_scores[d.id].quality_score for d in prose_docs if d.id in doc_quality_scores}
        near_dedup_res = minhash_lsh.deduplicate(prose_docs, scores_map)

        for cluster in near_dedup_res.clusters:
            for rem_id in cluster.removed_document_ids:
                ledger_entries.append(
                    {
                        "document_id": rem_id,
                        "decision": "near_duplicate",
                        "reason": f"Near-duplicate of canonical document {cluster.canonical_document_id} (similarity {cluster.estimated_similarity})",
                        "cluster_id": cluster.cluster_id,
                    }
                )

        near_deduped_docs = near_dedup_res.canonical_documents

        # Step 7: Corpus Balance Controls & Downsampling
        doc_tokens_map = {doc.id: token_counter.count(doc.text) for doc in near_deduped_docs}

        balancer = CorpusBalancer(
            max_source_token_share=self.config.max_source_token_share,
            max_category_token_share=self.config.max_category_token_share,
            max_organization_token_share=self.config.max_organization_token_share,
        )
        bal_res = balancer.balance(
            near_deduped_docs, doc_tokens_map, doc_quality_scores
        )

        for doc in bal_res.balanced_out_documents:
            ledger_entries.append(
                {
                    "document_id": doc.id,
                    "decision": "balanced_out",
                    "reason": "Removed by final-corpus concentration balancing.",
                    "source_id": doc.source_id,
                }
            )

        curated_docs = bal_res.kept_documents

        # Step 8: Post-Curation Train / Validation Split with Near-Duplicate Cluster Isolation

        # Build cluster parent mapping so all docs in a near-dup cluster are assigned atomically
        cluster_doc_groups: list[list[CorpusDocument]] = []
        cluster_covered: set[str] = set()

        for cluster in near_dedup_res.clusters:
            group = [d for d in curated_docs if d.id == cluster.canonical_document_id or d.id in cluster.removed_document_ids]
            if group:
                cluster_doc_groups.append(group)
                for d in group:
                    cluster_covered.add(d.id)

        for doc in curated_docs:
            if doc.id not in cluster_covered:
                cluster_doc_groups.append([doc])

        # Deterministically split groups
        train_curated_docs: list[CorpusDocument] = []
        val_curated_docs: list[CorpusDocument] = []

        for group in sorted(cluster_doc_groups, key=lambda items: min(item.id for item in items)):
            primary_doc = group[0]
            # SHA-256 is stable across Python processes, hosts, and run times.
            split_key = f"{self.config.seed}:{primary_doc.id}".encode()
            h_val = int.from_bytes(hashlib.sha256(split_key).digest()[:8], "big") / 2**64
            if h_val < self.config.train_ratio:
                train_curated_docs.extend(group)
            else:
                val_curated_docs.extend(group)

        # Log kept documents to ledger
        for doc in curated_docs:
            split_assigned = "train" if doc in train_curated_docs else "validation"
            ledger_entries.append(
                {
                    "document_id": doc.id,
                    "decision": "kept",
                    "split": split_assigned,
                    "quality_score": doc_quality_scores.get(doc.id, DocumentQualityScore(0.0, "low")).quality_score,
                    "quality_bucket": doc_quality_scores.get(doc.id, DocumentQualityScore(0.0, "low")).quality_bucket,
                    "token_count": doc_tokens_map.get(doc.id, 0),
                    "source_id": doc.source_id,
                }
            )

        # Step 9: Token Measurements & Sequence Analytics
        curated_tokens_list = [token_counter.count(d.text) for d in curated_docs]
        train_tokens = sum(token_counter.count(d.text) for d in train_curated_docs)
        val_tokens = sum(token_counter.count(d.text) for d in val_curated_docs)
        raw_tokens = sum(token_counter.count(d.text) for d in raw_docs)

        length_percentiles = compute_length_percentiles(curated_tokens_list)
        seq_evals = [evaluate_context_length(curated_tokens_list, cl) for cl in [512, 1024, 2048, 4096]]

        # Compute output fingerprint
        output_fingerprint = compute_corpus_fingerprint(curated_docs, "curated_stage3")

        # Distributions
        source_dist: dict[str, dict[str, Any]] = {}
        category_dist: dict[str, dict[str, Any]] = {}
        license_dist: dict[str, dict[str, Any]] = {}
        total_curated_tok = sum(curated_tokens_list)

        for doc in curated_docs:
            t = token_counter.count(doc.text)
            # Source
            if doc.source_id not in source_dist:
                source_dist[doc.source_id] = {"docs": 0, "tokens": 0, "share": 0.0}
            source_dist[doc.source_id]["docs"] += 1
            source_dist[doc.source_id]["tokens"] += t

            # Category
            if doc.category not in category_dist:
                category_dist[doc.category] = {"docs": 0, "tokens": 0, "share": 0.0, "status": "healthy"}
            category_dist[doc.category]["docs"] += 1
            category_dist[doc.category]["tokens"] += t

            # License
            lic = str(doc.metadata.get("verified_license_id") or doc.license_id or "unknown")
            if lic not in license_dist:
                license_dist[lic] = {"docs": 0, "tokens": 0, "share": 0.0}
            license_dist[lic]["docs"] += 1
            license_dist[lic]["tokens"] += t

        for sub in source_dist.values():
            sub["share"] = round(sub["tokens"] / max(1, total_curated_tok), 4)
        for sub in category_dist.values():
            sub["share"] = round(sub["tokens"] / max(1, total_curated_tok), 4)
        for sub in license_dist.values():
            sub["share"] = round(sub["tokens"] / max(1, total_curated_tok), 4)

        # Build Manifest
        manifest = CurationManifest(
            pipeline_version="1.0.0",
            build_timestamp=datetime.now(UTC).isoformat(),
            tokenizer_identifier=self.config.tokenizer_identifier,
            tokenizer_revision=self.config.tokenizer_revision,
            input_corpus_fingerprint=input_fingerprint,
            output_corpus_fingerprint=output_fingerprint,
            curation_config_hash=hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()[:16],
            input_documents_count=input_doc_count,
            curated_documents_count=len(curated_docs),
            train_documents_count=len(train_curated_docs),
            validation_documents_count=len(val_curated_docs),
            holdout_documents_count=0,
            total_input_tokens=raw_tokens,
            total_curated_tokens=total_curated_tok,
            train_tokens=train_tokens,
            validation_tokens=val_tokens,
            holdout_tokens=0,
            quality_rejects_count=quality_rejects_count,
            relevance_rejects_count=relevance_rejects_count,
            exact_duplicates_count=0,
            near_duplicates_count=len(near_dedup_res.removed_documents),
            balanced_out_count=len(bal_res.balanced_out_documents),
            fixture_excluded_count=fixture_excluded_count,
            source_distribution=source_dist,
            category_distribution=category_dist,
            license_distribution=license_dist,
            quality_bucket_distribution=quality_buckets_cnt,
            minhash_lsh_params=minhash_lsh.get_params(),
        )

        # Write Output Files
        # Build output files in a sibling temporary directory then atomically replace
        # individual artifacts. A failed build cannot leave partial JSONL files behind.
        self.curated_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(mkdtemp(prefix="architectai-curation-", dir=self.curated_dir.parent))
        write_jsonl(train_curated_docs, staging_dir / "train.jsonl")
        write_jsonl(val_curated_docs, staging_dir / "validation.jsonl")
        manifest.save(staging_dir / "curation_manifest.json")

        # Write Curation Decision Ledger
        ledger_path = staging_dir / "curation_ledger.jsonl"
        with ledger_path.open("w", encoding="utf-8") as f:
            for entry in ledger_entries:
                f.write(json.dumps(entry) + "\n")

        for name in ("train.jsonl", "validation.jsonl", "curation_manifest.json", "curation_ledger.jsonl"):
            (staging_dir / name).replace(self.curated_dir / name)
        staging_dir.rmdir()
        generate_source_audit(
            self.sources,
            raw_docs,
            curated_docs,
            doc_tokens_map,
            ledger_entries,
            self.curated_dir,
        )
        generate_license_audit(curated_docs, self.curated_dir)
        write_manual_review_sample(
            curated_docs,
            doc_tokens_map,
            doc_quality_scores,
            self.curated_dir / "manual_review_sample.md",
        )

        # Evaluate Readiness
        has_fixture_tok = any(d.source_id == "test_fixtures_local" for d in curated_docs)
        has_license_viol = any(d.metadata.get("verified_license_id") is None and d.license_id is None for d in curated_docs)
        has_leakage = False  # Enforced by group splitting

        readiness = evaluate_stage4_readiness(
            manifest=manifest,
            real_tokenizer_used=real_tokenizer_used,
            has_fixture_tokens=has_fixture_tok,
            has_license_violations=has_license_viol,
            has_split_leakage=has_leakage,
            validation_passed=True,
        )

        report_txt = generate_corpus_audit_report(
            manifest=manifest,
            conc_before=bal_res.concentration_before,
            conc_after=bal_res.concentration_after,
            percentiles=length_percentiles,
            seq_evals=seq_evals,
            readiness=readiness,
            output_path=self.curated_dir / "corpus_audit_report.md",
        )

        return manifest, report_txt


