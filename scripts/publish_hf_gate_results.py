#!/usr/bin/env python3
"""Publish a complete, immutable ArchitectAI benchmark result directory to HF."""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi

REQUIRED = (
    'quality_gate.json',
    'base/raw_outputs.jsonl', 'base/evaluations.jsonl', 'base/human_review.jsonl', 'base/baseline_manifest.json',
    'finetuned/raw_outputs.jsonl', 'finetuned/evaluations.jsonl', 'finetuned/human_review.jsonl', 'finetuned/baseline_manifest.json',
)

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--repo-id', default='babkindo/architectai-qwen3-8b-dapt-backups')
    args = parser.parse_args()
    results = args.results.resolve()
    missing = [name for name in REQUIRED if not (results / name).is_file()]
    if missing:
        raise SystemExit('incomplete benchmark results: ' + ', '.join(missing))
    manifest = {
        'result_directory': results.name,
        'published_at': datetime.now(timezone.utc).isoformat(),
        'files': {
            p.relative_to(results).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(results.rglob('*')) if p.is_file() and p.name not in {'gate.pid', 'gate.log'}
        },
    }
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type='model', private=True, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id, repo_type='model', folder_path=str(results),
        path_in_repo=f'evaluations/{results.name}',
        ignore_patterns=['gate.pid', 'gate.log'],
        commit_message=f'Publish complete {results.name} benchmark',
    )
    with tempfile.NamedTemporaryFile('w', suffix='.json', encoding='utf-8', delete=False) as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')
        manifest_path = Path(f.name)
    try:
        api.upload_file(repo_id=args.repo_id, repo_type='model', path_or_fileobj=str(manifest_path),
                        path_in_repo=f'manifests/{results.name}-results.json',
                        commit_message=f'Add {results.name} integrity manifest')
    finally:
        manifest_path.unlink(missing_ok=True)
    print(json.dumps({'status': 'published', 'results': results.name}))

if __name__ == '__main__':
    main()
