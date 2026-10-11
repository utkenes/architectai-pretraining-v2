#!/usr/bin/env python3
"""Publish complete ArchitectAI training checkpoints to a private Hugging Face repo."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO = "babkindo/architectai-qwen3-8b-dapt-backups"
REQUIRED = (
    "checkpoint_metadata.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "model/adapter_config.json",
    "model/adapter_model.safetensors",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_manifest(checkpoint: Path) -> dict[str, object]:
    missing = [name for name in REQUIRED if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete checkpoint {checkpoint}: missing {', '.join(missing)}")
    metadata = json.loads((checkpoint / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    files = {
        item.relative_to(checkpoint).as_posix(): {"bytes": item.stat().st_size, "sha256": sha256(item)}
        for item in sorted(checkpoint.rglob("*"))
        if item.is_file()
    }
    return {
        "checkpoint": checkpoint.name,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_metadata": metadata,
        "trainer_state": trainer_state,
        "files": files,
    }


def upload_checkpoint(api: HfApi, repo_id: str, checkpoint: Path) -> None:
    manifest = checkpoint_manifest(checkpoint)
    path_in_repo = f"checkpoints/{checkpoint.name}"
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(checkpoint),
        path_in_repo=path_in_repo,
        commit_message=f"Backup {checkpoint.name}",
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        manifest_path = Path(handle.name)
    try:
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=str(manifest_path),
            path_in_repo=f"manifests/{checkpoint.name}.json",
            commit_message=f"Add integrity manifest for {checkpoint.name}",
        )
    finally:
        manifest_path.unlink(missing_ok=True)


def upload_bootstrap(api: HfApi, repo_id: str, root: Path) -> None:
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    data_zip = root / "data/training/architectai_dapt_dataset_v3.zip"
    config = root / "configs/dapt.yaml"
    bootstrap = {
        "git_commit": git_sha,
        "dataset_zip": data_zip.name,
        "dataset_sha256": sha256(data_zip),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        manifest_path = temp / "reproducibility.json"
        manifest_path.write_text(json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        source_archive = temp / f"architectai-pretraining-v2-{git_sha[:12]}-source.tar.gz"
        subprocess.run(
            ["tar", "-czf", str(source_archive), "pyproject.toml", "src", "scripts", "configs", "tests/test_training_readiness.py"],
            cwd=root,
            check=True,
        )
        uploads = (
            (config, "reproducibility/configs/dapt.yaml"),
            (data_zip, f"reproducibility/dataset/{data_zip.name}"),
            (manifest_path, "reproducibility/reproducibility.json"),
            (source_archive, f"reproducibility/source/{source_archive.name}"),
        )
        for local, remote in uploads:
            api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_or_fileobj=str(local),
                path_in_repo=remote,
                commit_message=f"Add reproducibility artifact {Path(remote).name}",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-id", default=os.environ.get("ARCHITECTAI_HF_BACKUP_REPO", DEFAULT_REPO))
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    checkpoint = args.checkpoint.resolve()
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=True, exist_ok=True)
    if args.bootstrap:
        upload_bootstrap(api, args.repo_id, root)
    upload_checkpoint(api, args.repo_id, checkpoint)
    print(json.dumps({"repo_id": args.repo_id, "checkpoint": checkpoint.name, "status": "published"}))


if __name__ == "__main__":
    main()
