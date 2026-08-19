"""Run Stage 3 curation twice in independent interpreters and compare artifacts."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(output: Path, raw: str, manifest: str) -> dict[str, object]:
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src")
    command = [
        sys.executable, "-m", "architectai_pretraining", "curate",
        "--raw-accepted-dir", raw, "--manifest", manifest, "--curated-dir", str(output),
    ]
    subprocess.run(command, check=True, env=env)
    manifest_data = json.loads((output / "curation_manifest.json").read_text(encoding="utf-8"))
    train_ids = [json.loads(line)["id"] for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    validation_ids = [json.loads(line)["id"] for line in (output / "validation.jsonl").read_text(encoding="utf-8").splitlines()]
    return {
        "curated_fingerprint": manifest_data["output_corpus_fingerprint"],
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "train_tokens": manifest_data["train_tokens"],
        "validation_tokens": manifest_data["validation_tokens"],
        "source_distribution": manifest_data["source_distribution"],
        "category_distribution": manifest_data["category_distribution"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-accepted-dir", default="data/final/raw_accepted")
    parser.add_argument("--manifest", default="configs/sources.yaml")
    parser.add_argument("--output", default="scratch/reproducibility.json")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="architectai-repro-") as temporary:
        base = Path(temporary)
        first, second = _run(base / "first", args.raw_accepted_dir, args.manifest), _run(base / "second", args.raw_accepted_dir, args.manifest)
    result = {"identical": first == second, "first": first, "second": second}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"identical={str(result['identical']).lower()}")
    if not result["identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
