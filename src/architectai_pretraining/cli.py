"""Command Line Interface for ArchitectAI Domain Pretraining Corpus Pipeline."""

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from architectai_pretraining.curation import CurationConfig, CurationPipeline
from architectai_pretraining.diagnostics import SourceDiagnosticsEngine
from architectai_pretraining.io import read_jsonl
from architectai_pretraining.pipeline import CorpusPipeline, PipelineConfig
from architectai_pretraining.sources import load_source_manifest
from architectai_pretraining.stats import calculate_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="architectai-pretraining",
        description="Domain-adaptive pretraining (DAPT) corpus preparation tool",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Command: build
    build_parser = subparsers.add_parser("build", help="Build pretraining corpus from sources")
    build_parser.add_argument(
        "--manifest",
        default="configs/sources.yaml",
        help="Path to sources.yaml manifest (default: configs/sources.yaml)",
    )
    build_parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Path to raw source directory (default: data/raw)",
    )
    build_parser.add_argument(
        "--cache-dir",
        default=None,
        help="Path to Git repository cache directory (defaults to temp directory)",
    )
    build_parser.add_argument(
        "--final-dir",
        default="data/final",
        help="Output directory for final JSONL splits (default: data/final)",
    )
    build_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.98,
        help="Ratio of documents assigned to training split (default: 0.98)",
    )
    build_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split (default: 42)",
    )
    build_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview sources and candidate files without downloading or writing corpus outputs",
    )

    # Command: dry-run
    dry_parser = subparsers.add_parser(
        "dry-run", help="Preview source manifest and candidate ingestion files without downloading"
    )
    dry_parser.add_argument(
        "--manifest",
        default="configs/sources.yaml",
        help="Path to sources.yaml manifest (default: configs/sources.yaml)",
    )

    # Command: source-diagnostics
    diag_parser = subparsers.add_parser(
        "source-diagnostics", help="Diagnose catalog source ingestion status and zero-doc root causes"
    )
    diag_parser.add_argument(
        "--manifest", default="configs/sources.yaml", help="Path to sources.yaml manifest"
    )

    # Command: curate / prepare-training-corpus
    curate_parser = subparsers.add_parser(
        "curate", help="Run Stage 3 end-to-end corpus curation, balancing, and tokenization"
    )
    curate_parser.add_argument(
        "--manifest", default="configs/sources.yaml", help="Path to sources.yaml manifest"
    )
    curate_parser.add_argument(
        "--raw-accepted-dir", default="data/final/raw_accepted", help="Raw accepted input directory"
    )
    curate_parser.add_argument(
        "--curated-dir", default="data/final/curated", help="Curated output directory"
    )
    curate_parser.add_argument(
        "--tokenizer", default="Qwen/Qwen3-8B", help="Pinned Qwen family tokenizer identifier"
    )
    curate_parser.add_argument(
        "--allow-mock-fallback", action="store_true", help="Allow mock token counter fallback (test-only)"
    )

    subparsers.add_parser(
        "prepare-training-corpus",
        parents=[curate_parser],
        add_help=False,
        help="Alias for curate (runs Stage 3 pipeline)",
    )

    # Command: token-stats / audit
    subparsers.add_parser(
        "token-stats", help="Compute exact tokenizer counts across curated splits"
    )
    subparsers.add_parser(
        "audit", help="Generate pre/post curation audit reports"
    )

    # Command: sequence-analysis
    subparsers.add_parser(
        "sequence-analysis",
        help="Analyze document length percentiles and sequence packing efficiency",
    )

    # Command: validate-curated
    val_curated_parser = subparsers.add_parser(
        "validate-curated", help="Validate integrity of curated JSONL output splits"
    )
    val_curated_parser.add_argument(
        "--curated-dir", default="data/final/curated", help="Directory containing curated splits"
    )

    # Command: stats
    stats_parser = subparsers.add_parser(
        "stats", help="Inspect corpus statistics of raw final output"
    )
    stats_parser.add_argument(
        "--final-dir",
        default="data/final",
        help="Directory containing train.jsonl and validation.jsonl (default: data/final)",
    )

    # Command: validate
    val_parser = subparsers.add_parser(
        "validate", help="Validate integrity of exported JSONL splits"
    )
    val_parser.add_argument(
        "--final-dir",
        default="data/final",
        help="Directory containing train.jsonl and validation.jsonl (default: data/final)",
    )
    # Command: benchmark
    bench_parser = subparsers.add_parser(
        "benchmark", help="ArchitectAI Stage 4 Baseline Benchmark & Evaluation Harness"
    )
    bench_parser.add_argument(
        "action",
        nargs="?",
        default="baseline",
        choices=["validate", "contamination-check", "run", "report", "baseline", "gate"],
        help="Benchmark action to perform (default: baseline)",
    )
    bench_parser.add_argument(
        "--dataset", default="data/benchmark/architectai_v1.jsonl", help="Path to benchmark JSONL dataset"
    )
    bench_parser.add_argument(
        "--corpus-dir", default="data/final/curated", help="Path to training corpus for contamination checking"
    )
    bench_parser.add_argument(
        "--results-dir", default="data/benchmark/results/baseline", help="Directory to save benchmark results"
    )
    bench_parser.add_argument(
        "--config", default="configs/benchmark.yaml", help="Benchmark inference configuration"
    )
    bench_parser.add_argument(
        "--model", default=None, help="Base Qwen model identifier"
    )
    bench_parser.add_argument(
        "--tokenizer", default=None, help="Tokenizer identifier"
    )
    bench_parser.add_argument(
        "--quantization", default=None, choices=["none", "4bit", "8bit"], help="Inference quantization mode"
    )
    bench_parser.add_argument(
        "--device", default=None, help="Inference device (cuda/cpu)"
    )
    bench_parser.add_argument(
        "--adapter-path", default=None, help="PEFT adapter directory for a finetuned run or gate"
    )
    bench_parser.add_argument(
        "--checkpoint-metadata", default=None, help="Checkpoint metadata path; defaults next to adapter"
    )
    bench_parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=None,
        help="Override Qwen3 thinking mode (thinking requires sampling)",
    )
    bench_parser.add_argument("--max-new-tokens", type=int, default=None)
    bench_parser.add_argument(
        "--mock", action="store_true", help="Run using mock inference for infrastructure validation"
    )

    dapt_parser = subparsers.add_parser("dapt", help="Stage 5 DAPT preparation and Colab checks")
    dapt_actions = dapt_parser.add_subparsers(dest="dapt_action", required=True)
    dapt_actions.add_parser("preflight", help="Inspect GPU and dependency capability")
    dapt_verify = dapt_actions.add_parser("verify-data", help="Verify a Colab dataset package before model loading")
    dapt_verify.add_argument("--manifest", required=True, help="Path to checksum-verified dataset archive")
    dapt_package = dapt_actions.add_parser("package-data", help="Create checksum-verified curated dataset package")
    dapt_package.add_argument("--curated-dir", default="data/final/curated")
    dapt_package.add_argument("--output", default="data/training/architectai_dapt_dataset_v2.zip")
    dapt_package.add_argument("--build-git-sha", default=None, help="Optional explicit build SHA; defaults to git rev-parse HEAD")
    dapt_pack = dapt_actions.add_parser("pack", help="Prepare deterministic causal-LM train and validation JSONL")
    dapt_pack.add_argument("--curated-dir", default="data/final/curated")
    dapt_pack.add_argument("--output-dir", default="data/training/packed")
    dapt_pack.add_argument("--sequence-length", type=int, default=2048)
    dapt_readiness = dapt_actions.add_parser("readiness", help="Generate Stage 4.1 GO/NO-GO report")
    dapt_readiness.add_argument("--curated-dir", default="data/final/curated")
    dapt_readiness.add_argument("--output-dir", default="data/training")
    dapt_final_readiness = dapt_actions.add_parser("final-readiness", help="Generate Stage 4.2 final report")
    dapt_final_readiness.add_argument("--curated-dir", default="data/final/curated")
    dapt_final_readiness.add_argument("--output-dir", default="data/training")
    dapt_final_readiness.add_argument("--archive", default="data/training/architectai_dapt_dataset_v2.zip")
    dapt_smoke = dapt_actions.add_parser("smoke", help="Show guarded Stage 5A smoke execution requirements")
    dapt_smoke.add_argument("--max-steps", type=int, default=20)
    dapt_smoke.add_argument("--config", default="configs/dapt.yaml")
    dapt_smoke.add_argument("--output-dir", default=None)
    dapt_smoke.add_argument("--resume-from", default=None)

    args = parser.parse_args()

    if args.command == "dapt":
        if args.dapt_action == "preflight":
            from architectai_pretraining.training.preflight import inspect_environment

            print(json.dumps(inspect_environment(), indent=2))
        elif args.dapt_action == "verify-data":
            from architectai_pretraining.training.package import verify_dataset_package

            print(json.dumps(verify_dataset_package(args.manifest), indent=2))
        elif args.dapt_action == "package-data":
            from architectai_pretraining.training.package import create_dataset_package

            print(json.dumps(create_dataset_package(args.curated_dir, args.output, args.build_git_sha), indent=2))
        elif args.dapt_action == "pack":
            from architectai_pretraining.tokenizer import HuggingFaceTokenCounter
            from architectai_pretraining.training.data import pack_documents

            tokenizer = HuggingFaceTokenCounter()
            curated = Path(args.curated_dir)
            output = Path(args.output_dir)
            for split in ("train", "validation"):
                packed = pack_documents(read_jsonl(curated / f"{split}.jsonl"), tokenizer, args.sequence_length)
                packed.write_jsonl(output / f"{split}.jsonl")
                packed.write_manifest(output / f"{split}_manifest.json")
                print(f"{split}: {packed.statistics.sequence_count} sequences; fingerprint={packed.fingerprint}")
        elif args.dapt_action == "readiness":
            from architectai_pretraining.training.readiness import generate_readiness_report

            result = generate_readiness_report(args.curated_dir, args.output_dir)
            for key in ("READY_FOR_COLAB_BASELINE", "READY_FOR_STAGE_5A_SMOKE", "GO_FOR_FULL_DAPT"):
                print(f"{key}={str(result[key]).lower()}")
            if result["blocking_issues"]:
                print("Blocking reasons:")
                for reason in result["blocking_issues"]:
                    print(f"- {reason}")
        elif args.dapt_action == "final-readiness":
            from architectai_pretraining.training.final_readiness import (
                generate_final_readiness_report,
            )

            result = generate_final_readiness_report(args.curated_dir, args.output_dir, args.archive)
            for key in ("READY_FOR_COLAB", "READY_TO_RUN_REAL_BASELINE", "READY_FOR_STAGE_5A_REAL_SMOKE", "GO_FOR_FULL_DAPT"):
                print(f"{key}={str(result[key]).lower()}")
        else:
            if args.max_steps < 1 or args.max_steps > 50:
                raise ValueError("Stage 5A smoke must use 1-50 steps.")
            from architectai_pretraining.manifest import CurationManifest
            from architectai_pretraining.training.runner import (
                load_smoke_config,
                run_smoke_training,
            )

            smoke_config = load_smoke_config(args.config, args.output_dir)
            smoke_config = dataclasses.replace(smoke_config, max_steps=args.max_steps)
            curated = Path("data/final/curated")
            manifest = CurationManifest(**json.loads((curated / "curation_manifest.json").read_text(encoding="utf-8")))
            train_packed = json.loads((Path("data/training/packed") / "train_manifest.json").read_text(encoding="utf-8"))
            validation_packed = json.loads((Path("data/training/packed") / "validation_manifest.json").read_text(encoding="utf-8"))
            result = run_smoke_training(smoke_config, manifest.output_corpus_fingerprint, train_packed["fingerprint"], validation_packed["fingerprint"], args.resume_from)
            print(json.dumps(result, indent=2))
        return

    if args.command == "benchmark":
        from architectai_pretraining.benchmark.contamination import check_benchmark_against_corpus
        from architectai_pretraining.benchmark.dataset import load_benchmark_dataset
        from architectai_pretraining.benchmark.gate import run_base_vs_finetuned_gate
        from architectai_pretraining.benchmark.models import InferenceConfig
        from architectai_pretraining.benchmark.report import generate_benchmark_reports
        from architectai_pretraining.benchmark.runner import BenchmarkRunner

        dataset_path = Path(args.dataset)
        results_path = Path(args.results_dir)
        corpus_path = Path(args.corpus_dir)

        print(f"Loading ArchitectAI Benchmark dataset from {dataset_path}...")
        dataset = load_benchmark_dataset(dataset_path)

        cat_dist = dataset.get_category_distribution()
        diff_dist = dataset.get_difficulty_distribution()

        print("  Benchmark Version:    architectai-bench-v1")
        print(f"  Benchmark Fingerprint: {dataset.fingerprint}")
        print(f"  Total Scenarios:      {len(dataset.samples)}")
        print(f"  Difficulty Specs:     {diff_dist}")
        print(f"  Categories Count:     {len(cat_dist)}")
        print()

        if args.action in ("contamination-check", "baseline"):
            print(f"Checking benchmark contamination against corpus directory '{corpus_path}'...")
            contam_res = check_benchmark_against_corpus(dataset, corpus_path)
            print(f"  Contamination Rate:   {contam_res.contamination_rate:.2%}")
            print(f"  Contaminated Items:   {contam_res.contaminated_scenarios}")
            if contam_res.contaminated_scenarios > 0:
                print("  WARNING: Contaminated scenarios detected!")
                for item in contam_res.flagged_items:
                    print(f"    - {item['sample_id']} <-> {item['corpus_doc_id']}: {item['reason']}")
            else:
                print("  Clean: Zero benchmark contamination found in training corpus.")
            print()

        if args.action in ("run", "baseline", "gate"):
            benchmark_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
            model_config = benchmark_config["model"]
            generation_config = benchmark_config["generation"]
            inf_config = InferenceConfig(
                model_identifier=args.model or model_config["identifier"],
                revision=model_config["revision"],
                tokenizer_identifier=args.tokenizer or model_config["tokenizer_identifier"],
                quantization=args.quantization or model_config["quantization"],
                dtype=model_config["dtype"],
                device=args.device or model_config["device"],
                temperature=float(generation_config["temperature"]),
                do_sample=bool(generation_config["do_sample"]),
                enable_thinking=(
                    bool(generation_config["enable_thinking"])
                    if args.enable_thinking is None else args.enable_thinking
                ),
                max_new_tokens=args.max_new_tokens or int(generation_config["max_new_tokens"]),
            )

            if args.action == "gate":
                if args.mock:
                    raise ValueError("The base-vs-finetuned behavior gate cannot run in mock mode.")
                if not args.adapter_path:
                    raise ValueError("benchmark gate requires --adapter-path.")
                gate_result = run_base_vs_finetuned_gate(dataset, inf_config, args.adapter_path, results_path)
                print(json.dumps(gate_result.to_dict(), indent=2))
                print("DAPT_BEHAVIOR_GATE=true")
                return

            inf_config = dataclasses.replace(
                inf_config,
                adapter_path=args.adapter_path,
                checkpoint_metadata_path=args.checkpoint_metadata,
            )

            print(f"Running benchmark inference (Mock={args.mock}, Model='{inf_config.model_identifier}', Quantization='{inf_config.quantization}')...")
            runner = BenchmarkRunner(
                dataset=dataset,
                config=inf_config,
                results_dir=results_path,
                use_mock=args.mock,
            )
            bench_manifest, evaluations = runner.run()

            raw_outputs = list(runner._load_completed_raw_outputs().values())
            report_p, failure_p = generate_benchmark_reports(dataset, evaluations, raw_outputs, bench_manifest, results_path)

            print("\n==================================================")
            print("    Stage 4 Baseline Benchmark Complete            ")
            print("==================================================")
            print(f"Completed Scenarios: {bench_manifest.completed_count}/{bench_manifest.scenario_count}")
            print(f"Mock Execution Mode: {bench_manifest.is_mock_run}")
            print(f"Baseline Manifest:   {results_path}/baseline_manifest.json")
            print(f"Main Report:         {report_p}")
            print(f"Failure Modes Analysis: {failure_p}")
            print(f"Human Review File:   {results_path}/human_review.jsonl")
            print("--------------------------------------------------")
            if bench_manifest.ready_for_stage_5:
                print("READY_FOR_STAGE_5_DAPT=true")
            else:
                print("READY_FOR_STAGE_5_DAPT=false (Mock mode or incomplete execution)")

    if args.command == "dry-run" or (args.command == "build" and getattr(args, "dry_run", False)):
        manifest_path = getattr(args, "manifest", "configs/sources.yaml")
        cache_dir = getattr(args, "cache_dir", None)

        kwargs: dict[str, Any] = {"manifest_path": manifest_path}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir

        config = PipelineConfig(**kwargs)
        pipeline = CorpusPipeline(config)
        preview = pipeline.dry_run()

        print("==================================================")
        print("    ArchitectAI Corpus Pipeline Ingestion Dry-Run ")
        print("==================================================")
        print(f"Total Sources Configured: {preview['total_sources']}")
        print(f"Enabled Sources:          {preview['enabled_sources']}")
        print(f"Disabled Sources:         {preview['disabled_sources']}")
        print(f"Total Candidate Files:    {preview['total_candidate_files']}")
        print("--------------------------------------------------")

        for s in preview["sources"]:
            status = "ENABLED" if s["enabled"] else "DISABLED"
            lic = s["license_id"] or "Unverified/None"
            print(f"Source ID: {s['id']} [{status}]")
            print(f"  Name:       {s['name']}")
            print(f"  Category:   {s['category']}")
            print(f"  Type:       {s['type']}")
            print(f"  License:    {lic}")
            print(f"  URL/Path:   {s['url'] or s['path']}")
            if s["enabled"]:
                print(f"  Candidates: {s['candidate_count']} files")
            print()
        print("Dry-run complete. No files downloaded or modified.")
        sys.exit(0)

    elif args.command == "source-diagnostics":
        manifest_path = Path(getattr(args, "manifest", "configs/sources.yaml"))
        sources = load_source_manifest(manifest_path)
        from tempfile import gettempdir
        cache_p = Path(gettempdir()) / "architectai_git_cache"
        engine = SourceDiagnosticsEngine(sources, cache_p)
        res = engine.generate_report(Path("data/final/curated/source_diagnostics.json"))
        print("==================================================")
        print("     ArchitectAI Catalog Source Diagnostics       ")
        print("==================================================")
        print(f"Total Catalog Sources:  {res['total_catalog_sources']}")
        print(f"Active Sources:         {res['active_sources_count']}")
        print(f"Disabled Sources:       {res['disabled_sources_count']}")
        print("--------------------------------------------------")
        for d in res["diagnostics"]:
            st = d["status"].upper()
            print(f"[{st}] {d['source_id']} ({d['name']})")
            print(f"  Category:       {d['category']}")
            print(f"  Classification: {d['classification']}")
            print(f"  Explanation:    {d['explanation']}")
            print()

    elif args.command in ("curate", "prepare-training-corpus"):
        manifest_path = getattr(args, "manifest", "configs/sources.yaml")
        raw_dir = getattr(args, "raw_accepted_dir", "data/final/raw_accepted")
        curated_dir = getattr(args, "curated_dir", "data/final/curated")
        tok_id = getattr(args, "tokenizer", "Qwen/Qwen3-8B")
        allow_mock = getattr(args, "allow_mock_fallback", False)

        c_config = CurationConfig(
            manifest_path=manifest_path,
            raw_accepted_dir=raw_dir,
            curated_dir=curated_dir,
            tokenizer_identifier=tok_id,
            fallback_allowed_in_prod=allow_mock,
        )
        cur_pipeline = CurationPipeline(c_config)
        print("Starting ArchitectAI Stage 3 Corpus Curation Pipeline...\n")
        manifest, report_text = cur_pipeline.run()

        print("==================================================")
        print("    Stage 3 Curation Pipeline Execution Complete  ")
        print("==================================================")
        print(f"Input Raw Documents:       {manifest.input_documents_count:,}")
        print(f"Curated Output Documents:  {manifest.curated_documents_count:,}")
        print(f"Total Curated Tokens:      {manifest.total_curated_tokens:,}")
        print(f"Train Split Tokens:        {manifest.train_tokens:,}")
        print(f"Validation Split Tokens:   {manifest.validation_tokens:,}")
        print(f"Output Corpus Fingerprint: {manifest.output_corpus_fingerprint}")
        print("--------------------------------------------------")
        print(f"Curation Manifest: {curated_dir}/curation_manifest.json")
        print(f"Audit Report:      {curated_dir}/corpus_audit_report.md")

    elif args.command in ("validate-curated", "validate"):
        cur_dir = Path(getattr(args, "curated_dir", getattr(args, "final_dir", "data/final/curated")))
        train_p = cur_dir / "train.jsonl"
        val_p = cur_dir / "validation.jsonl"

        if not train_p.exists():
            cur_dir = Path("data/final")
            train_p = cur_dir / "train.jsonl"
            val_p = cur_dir / "validation.jsonl"

        total = 0
        for p in [train_p, val_p]:
            if p.exists():
                docs = read_jsonl(p)
                total += len(docs)
                print(f"Validated {len(docs)} documents in {p.name} successfully.")

        if total > 0:
            print("\nValidation PASSED: All corpus records are valid CorpusDocument objects.")
        else:
            print("\nValidation FAILED: No JSONL records found.")
            sys.exit(1)

    elif args.command == "build" or args.command is None:
        manifest_path = getattr(args, "manifest", "configs/sources.yaml")
        final_dir = getattr(args, "final_dir", "data/final")
        cache_dir = getattr(args, "cache_dir", None)
        train_ratio = getattr(args, "train_ratio", 0.98)
        seed = getattr(args, "seed", 42)

        build_kwargs: dict[str, Any] = {
            "manifest_path": manifest_path,
            "final_dir": final_dir,
            "train_ratio": train_ratio,
            "seed": seed,
        }
        if cache_dir is not None:
            build_kwargs["cache_dir"] = cache_dir

        config = PipelineConfig(**build_kwargs)
        pipeline = CorpusPipeline(config)

        print("Starting ArchitectAI Domain Pretraining Corpus Build...\n")
        stats, train_path, val_path = pipeline.run()

        print(stats.to_formatted_report())
        print("\nCorpus build complete.")
        print(f"  Train Output:      {train_path}")
        print(f"  Validation Output: {val_path}")

    elif args.command == "stats":
        final_dir_path = Path(args.final_dir)
        train_path = final_dir_path / "train.jsonl"
        val_path = final_dir_path / "validation.jsonl"

        train_docs = read_jsonl(train_path) if train_path.exists() else []
        val_docs = read_jsonl(val_path) if val_path.exists() else []

        if not train_docs and not val_docs:
            print(f"No corpus files found in {final_dir_path}. Run 'build' first.")
            sys.exit(1)

        stats = calculate_stats(train_docs, val_docs)
        print(stats.to_formatted_report())


if __name__ == "__main__":
    main()


# Qwen3 update
