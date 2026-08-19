"""Tests for source ingestion adapters, pattern matching, and strict license verifier."""

import pytest

from architectai_pretraining.sources import (
    LocalDirectorySourceAdapter,
    SourceConfig,
    detect_spdx_license,
    get_adapter,
    load_source_manifest,
    matches_patterns,
    verify_repository_license,
)


def test_load_source_manifest(tmp_path: pytest.TempPathFactory) -> None:
    manifest_content = """
sources:
  - id: test_src
    name: "Test Source"
    category: "architecture_patterns"
    enabled: true
    type: "local_directory"
    path: "data/raw"
    license_id: "MIT"
"""
    manifest_file = tmp_path / "sources.yaml"  # type: ignore[operator]
    manifest_file.write_text(manifest_content, encoding="utf-8")

    configs = load_source_manifest(manifest_file)
    assert len(configs) == 1
    assert configs[0].id == "test_src"
    assert configs[0].license_id == "MIT"


def test_detect_spdx_license() -> None:
    mit_text = "Permission is hereby granted, free of charge, under the MIT License..."
    apache_text = "Licensed under the Apache License, Version 2.0..."
    cc_by_text = "Creative Commons Attribution 4.0 International Public License (CC-BY-4.0)..."
    cc_sa_text = "Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0)..."

    assert detect_spdx_license(mit_text) == "MIT"
    assert detect_spdx_license(apache_text) == "Apache-2.0"
    assert detect_spdx_license(cc_by_text) == "CC-BY-4.0"
    assert detect_spdx_license(cc_sa_text) == "CC-BY-SA-4.0"


def test_verify_repository_license_match(tmp_path: pytest.TempPathFactory) -> None:
    repo_dir = tmp_path / "mock_repo"  # type: ignore[operator]
    repo_dir.mkdir()
    lic_file = repo_dir / "LICENSE"
    lic_file.write_text("MIT License\n\nPermission is hereby granted...", encoding="utf-8")

    res = verify_repository_license(repo_dir, "MIT")
    assert res.is_valid is True
    assert res.verified_license_id == "MIT"
    assert res.license_source == "detected_file:LICENSE"


def test_verify_repository_license_mismatch_fails(tmp_path: pytest.TempPathFactory) -> None:
    repo_dir = tmp_path / "mock_repo_mismatch"  # type: ignore[operator]
    repo_dir.mkdir()
    lic_file = repo_dir / "LICENSE"
    lic_text = "Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0)"
    lic_file.write_text(lic_text, encoding="utf-8")

    # Manifest claims CC-BY-4.0, but repo contains CC-BY-SA-4.0
    res = verify_repository_license(repo_dir, "CC-BY-4.0")
    assert res.is_valid is False
    assert "License mismatch!" in (res.error_message or "")


def test_matches_patterns(tmp_path: pytest.TempPathFactory) -> None:
    base = tmp_path / "base"  # type: ignore[operator]
    base.mkdir()
    doc1 = base / "docs" / "arch.md"
    doc1.parent.mkdir(parents=True, exist_ok=True)
    doc1.write_text("# Doc 1", encoding="utf-8")

    ignored = base / "node_modules" / "pkg.md"
    ignored.parent.mkdir(parents=True, exist_ok=True)
    ignored.write_text("# Ignored", encoding="utf-8")

    include = ["docs/**/*.md"]
    exclude = ["node_modules/**"]

    assert matches_patterns(doc1, base, include, exclude) is True
    assert matches_patterns(ignored, base, include, exclude) is False


def test_local_file_source_adapter(tmp_path: pytest.TempPathFactory) -> None:
    test_file = tmp_path / "arch_doc.md"  # type: ignore[operator]
    test_content = "# High Availability Architecture\n\nDetails on active-passive failover."
    test_file.write_text(test_content, encoding="utf-8")

    cfg = SourceConfig(
        id="single_file_src",
        name="Single File Source",
        category="reliability",
        type="local_file",
        path=str(test_file),
        license_id="CC-BY-4.0",
    )

    adapter = get_adapter(cfg)
    docs = adapter.ingest()

    assert len(docs) == 1
    assert docs[0].title == "High Availability Architecture"
    assert docs[0].metadata["license_source"] == "declared_manifest"
    assert docs[0].metadata["verified_license_id"] == "CC-BY-4.0"


def test_local_source_ingests_asciidoc_with_heading(tmp_path: pytest.TempPathFactory) -> None:
    docs_dir = tmp_path / "docs"  # type: ignore[operator]
    docs_dir.mkdir()
    (docs_dir / "resilience.adoc").write_text(
        "= Circuit Breaker\n\nA circuit breaker protects a distributed system.", encoding="utf-8"
    )
    config = SourceConfig(
        id="resilience", name="Resilience", category="reliability", type="local_directory",
        path=str(docs_dir), license_id="Apache-2.0", include_patterns=["*.adoc"],
    )
    documents = LocalDirectorySourceAdapter(config).ingest()
    assert documents[0].title == "Circuit Breaker"
    assert documents[0].metadata["relative_path"] == "resilience.adoc"


def test_local_html_chapter_extraction_removes_navigation(tmp_path: pytest.TempPathFactory) -> None:
    docs_dir = tmp_path / "html_docs"  # type: ignore[operator]
    raw = docs_dir / "raw"
    raw.mkdir(parents=True)
    (raw / "ch01.html").write_text(
        "<nav>Table of contents</nav><h1>Retries</h1><p>Retries need timeout budgets.</p>"
        "<script>ignored()</script><ul><li>Bound attempts</li></ul>",
        encoding="utf-8",
    )
    config = SourceConfig(
        id="html", name="HTML", category="reliability", type="local_directory", path=str(docs_dir),
        license_id="CC-BY-4.0", parser="html", allowed_content_types=["html"],
        include_patterns=["raw/ch*.html"],
    )
    document = LocalDirectorySourceAdapter(config).ingest()[0]
    assert "# Retries" in document.text
    assert "Retries need timeout budgets." in document.text
    assert "Table of contents" not in document.text
    assert "ignored" not in document.text
