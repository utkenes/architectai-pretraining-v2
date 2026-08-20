"""Raw source ingestion adapters, YAML manifest loader, and strict license verifier."""

import fnmatch
import hashlib
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests
import yaml

from architectai_pretraining.models import CorpusDocument

logger = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    """Configuration definition for a source defined in configs/sources.yaml."""

    id: str
    name: str
    category: str
    enabled: bool = True
    type: str = "local_directory"
    path: str | None = None
    url: str | None = None
    license_id: str | None = None
    language: str = "en"
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None
    allowed_content_types: list[str] | None = None
    parser: str | None = None
    source_token_cap: int | None = None
    source_priority: float = 1.0
    notes: str | None = None
    verify_license: bool = False
    allow_unverified_license: bool = False
    license_training_status: str = "unverified"
    release_eligible: bool = False
    license_review_status: str = "needs_manual_review"
    license_evidence_path: str | None = None
    license_policy: dict[str, Any] = field(default_factory=dict)
    strip_section_patterns: list[str] | None = None
    section_category_rules: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LicenseVerificationResult:
    """Result of exact document/repository level license verification."""

    is_valid: bool
    declared_license_id: str | None
    detected_license_id: str | None
    verified_license_id: str | None
    license_source: str
    error_message: str | None = None


def detect_spdx_license(text: str) -> str | None:
    """Identify SPDX license identifier from license text contents."""
    text_strip = text.strip().upper()
    text_lower = text.lower()

    if "MIT OR CC0" in text_strip or "CC0 OR MIT" in text_strip:
        return "MIT"

    if "mit no attribution" in text_lower or "mit-0" in text_lower:
        return "MIT-0"

    # Check Apache before generic prose below: Apache 2.0's legal text can
    # mention third-party/Creative Commons material in a notice.
    if "apache license" in text_lower and ("version 2.0" in text_lower or "v2.0" in text_lower):
        return "Apache-2.0"

    # Creative Commons licenses - check ShareAlike & NonCommercial specifically before standard BY
    if "creative commons" in text_lower or "cc-" in text_lower or "attribution" in text_lower:
        if "noncommercial" in text_lower or "by-nc" in text_lower or "nc-sa" in text_lower:
            return "CC-BY-NC-SA-4.0"
        if (
            "attribution-sharealike 4.0" in text_lower
            or "by-sa/4.0" in text_lower
            or "by-sa 4.0" in text_lower
            or "cc-by-sa-4.0" in text_lower
        ):
            return "CC-BY-SA-4.0"
        if (
            "attribution 4.0" in text_lower
            or "by/4.0" in text_lower
            or "by 4.0" in text_lower
            or "cc-by-4.0" in text_lower
        ):
            return "CC-BY-4.0"
        if "attribution 3.0" in text_lower or "by/3.0" in text_lower or "by 3.0" in text_lower:
            return "CC-BY-3.0"
        if "public domain" in text_lower or "cc0" in text_lower:
            return "CC0-1.0"

    if "mit license" in text_lower or "permission is hereby granted, free of charge" in text_lower or text_strip == "MIT":
        return "MIT"

    if "bsd 3-clause" in text_lower or (
        "redistribution and use in source" in text_lower and "neither the name" in text_lower
    ):
        return "BSD-3-Clause"

    if "bsd 2-clause" in text_lower:
        return "BSD-2-Clause"

    if "mozilla public license" in text_lower and "2.0" in text_lower:
        return "MPL-2.0"

    return None


def verify_repository_license(
    repo_dir: Path, declared_license_id: str | None
) -> LicenseVerificationResult:
    """Verify license at exact repository level by examining root LICENSE file contents."""
    license_files = [
        "LICENSE",
        "LICENSE.md",
        "LICENSE.txt",
        "LICENCE",
        "LICENCE.md",
        "LICENCE.txt",
        "COPYING",
        "LICENSE.rst",
        "LICENSE-2.0.txt",
        "LICENSE-APACHE",
        "LICENSE-MIT",
        "LICENSE.MIT",
        "LICENSE.CC0-1.0",
    ]
    found_file: Path | None = None
    for filename in license_files:
        p = repo_dir / filename
        if p.is_file():
            found_file = p
            break

    if not found_file:
        return LicenseVerificationResult(
            is_valid=False,
            declared_license_id=declared_license_id,
            detected_license_id=None,
            verified_license_id=None,
            license_source="none",
            error_message=f"No LICENSE file found in repository root at '{repo_dir}'.",
        )

    try:
        text = found_file.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return LicenseVerificationResult(
            is_valid=False,
            declared_license_id=declared_license_id,
            detected_license_id=None,
            verified_license_id=None,
            license_source=f"detected_file:{found_file.name}",
            error_message=f"Error reading LICENSE file: {e}",
        )

    detected_spdx = detect_spdx_license(text)

    if not declared_license_id:
        return LicenseVerificationResult(
            is_valid=False,
            declared_license_id=None,
            detected_license_id=detected_spdx,
            verified_license_id=None,
            license_source=f"detected_file:{found_file.name}",
            error_message="Source manifest lacks explicit declared license_id.",
        )

    if detected_spdx is None:
        return LicenseVerificationResult(
            is_valid=False,
            declared_license_id=declared_license_id,
            detected_license_id=None,
            verified_license_id=None,
            license_source=f"detected_file:{found_file.name}",
            error_message=(
                f"LICENSE file content in '{found_file.name}' could not be matched "
                "to a known SPDX license."
            ),
        )

    decl_norm = declared_license_id.strip().upper()
    det_norm = detected_spdx.strip().upper()

    if decl_norm != det_norm:
        return LicenseVerificationResult(
            is_valid=False,
            declared_license_id=declared_license_id,
            detected_license_id=detected_spdx,
            verified_license_id=None,
            license_source=f"detected_file:{found_file.name}",
            error_message=(
                f"License mismatch! Manifest declared '{declared_license_id}' "
                f"but detected '{detected_spdx}' in '{found_file.name}'."
            ),
        )

    return LicenseVerificationResult(
        is_valid=True,
        declared_license_id=declared_license_id,
        detected_license_id=detected_spdx,
        verified_license_id=detected_spdx,
        license_source=f"detected_file:{found_file.name}",
        error_message=None,
    )


def matches_patterns(
    filepath: Path,
    base_dir: Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> bool:
    """Check if a file matches inclusion/exclusion wildcards relative to base_dir."""
    try:
        rel_path = filepath.relative_to(base_dir)
        rel_str = rel_path.as_posix()
    except ValueError:
        rel_path = filepath
        rel_str = filepath.as_posix()

    name = filepath.name

    norm_str = str(filepath).replace("\\", "/").lower()
    if "data/benchmark" in norm_str or "benchmark/" in norm_str:
        return False

    if exclude_patterns:
        for pat in exclude_patterns:
            if fnmatch.fnmatch(rel_str, pat) or fnmatch.fnmatch(name, pat) or rel_path.match(pat):
                return False

    if include_patterns:
        for pat in include_patterns:
            if fnmatch.fnmatch(rel_str, pat) or fnmatch.fnmatch(name, pat) or rel_path.match(pat):
                return True
            if "**" in pat:
                simplified = pat.replace("**/", "*").replace("/**", "*")
                if fnmatch.fnmatch(rel_str, simplified):
                    return True
        return False

    return filepath.suffix.lower() in {".md", ".markdown", ".txt", ".adoc", ".rst"}


def extract_sparse_checkout_dirs(include_patterns: list[str] | None) -> list[str]:
    """Extract top-level directory names from include patterns for git sparse checkout."""
    # Root-level patterns (for example ``*.adoc``) cannot be represented by a
    # directory-only sparse checkout. Use a complete checkout so source recovery
    # cannot silently discard valid root or module documentation.
    if include_patterns and any("/" not in pattern for pattern in include_patterns):
        return []
    sparse_dirs: set[str] = set()
    if include_patterns:
        for pat in include_patterns:
            parts = pat.split("/")
            if parts[0] and "*" not in parts[0]:
                sparse_dirs.add(parts[0])
    return sorted(sparse_dirs)


class BaseSourceAdapter(ABC):
    """Abstract base class for data ingestion adapters."""

    def __init__(self, config: SourceConfig, cache_dir: str | Path | None = None) -> None:
        self.config = config
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            default_cache = os.getenv(
                "ARCHITECTAI_CACHE_DIR",
                str(Path(tempfile.gettempdir()) / "architectai_git_cache"),
            )
            self.cache_dir = Path(default_cache)

    @abstractmethod
    def ingest(self) -> list[CorpusDocument]:
        """Ingest raw documents and return candidate CorpusDocument objects."""
        pass

    def list_candidate_files(self) -> list[Path]:
        """List target files matching source rules without processing content (dry-run)."""
        return []


class LocalDirectorySourceAdapter(BaseSourceAdapter):
    """Ingests Markdown (.md) and plain text (.txt) files from a local directory."""

    def list_candidate_files(self) -> list[Path]:
        if not self.config.path:
            return []

        dir_path = _resolve_configured_path(self.config.path)

        if not dir_path.exists() or not dir_path.is_dir():
            return []

        filepaths: list[Path] = []
        for p in dir_path.rglob("*"):
            if p.is_file() and matches_patterns(
                p, dir_path, self.config.include_patterns, self.config.exclude_patterns
            ) and _matches_allowed_content_type(p, self.config.allowed_content_types):
                filepaths.append(p)

        return sorted(filepaths)

    def ingest(self) -> list[CorpusDocument]:
        if not self.config.path:
            raise ValueError(
                f"Source '{self.config.id}' of type {self.config.type} requires 'path'"
            )

        dir_path = _resolve_configured_path(self.config.path)

        verification = _verify_local_source_license(dir_path, self.config)
        if verification is not None and not verification.is_valid and not self.config.allow_unverified_license:
            logger.warning("[LICENSE REJECTED] Source '%s': %s", self.config.id, verification.error_message)
            return []

        filepaths = self.list_candidate_files()
        documents: list[CorpusDocument] = []

        for filepath in filepaths:
            content_license = _resolve_content_license(filepath, dir_path, self.config)
            if not content_license["training_enabled"]:
                continue
            try:
                content = _read_source_text(filepath, self.config.parser)
            except Exception:
                content = _read_source_text(filepath, self.config.parser, errors="replace")

            if not content.strip():
                continue

            rel_path = filepath.relative_to(dir_path).as_posix()
            doc_id_input = f"{self.config.id}:{rel_path}"
            doc_id = hashlib.sha256(doc_id_input.encode("utf-8")).hexdigest()[:16]

            title = _extract_markdown_title(content, filepath.stem)
            verified_license_id = content_license["license_id"] or (
                verification.verified_license_id
                if verification is not None and verification.is_valid
                else self.config.license_id
            )

            doc = CorpusDocument(
                id=doc_id,
                source_id=self.config.id,
                source_url=self.config.url or f"file://{filepath.as_posix()}",
                license_id=verified_license_id,
                category=self.config.category,
                title=title,
                text=content,
                language=self.config.language,
                metadata={
                    "file_name": filepath.name,
                    "relative_path": rel_path,
                    "source_name": self.config.name,
                    "source_path": str(filepath),
                    "license_source": verification.license_source if verification else "declared_manifest",
                    "verified_license_id": verified_license_id,
                    "license_policy_type": content_license["policy_type"],
                    "license_evidence_path": content_license["evidence_path"],
                    "content_license_id": content_license["license_id"],
                    "license_verified": bool(verification and verification.is_valid)
                    or content_license["policy_type"] == "path_scoped",
                    "license_training_status": self.config.license_training_status,
                    "license_review_status": self.config.license_review_status,
                    "release_eligible": self.config.release_eligible,
                    "verified_commit_sha": "local",
                    **self.config.metadata,
                },
                source_name=self.config.name,
                source_path=str(filepath),
                relative_path=rel_path,
                verified_license_id=verified_license_id,
                source_priority=self.config.source_priority,
            )
            documents.append(doc)

        return documents


class MarkdownDirectorySourceAdapter(LocalDirectorySourceAdapter):
    """Adapter specifically for Markdown documentation directories."""

    pass


class LocalFileSourceAdapter(BaseSourceAdapter):
    """Ingests a single local text or Markdown file."""

    def list_candidate_files(self) -> list[Path]:
        if not self.config.path:
            return []
        filepath = Path(self.config.path)
        if not filepath.is_absolute():
            filepath = Path.cwd() / filepath
        return [filepath] if filepath.is_file() else []

    def ingest(self) -> list[CorpusDocument]:
        if not self.config.path:
            raise ValueError(f"Source '{self.config.id}' requires 'path'")

        filepaths = self.list_candidate_files()
        if not filepaths:
            return []

        filepath = filepaths[0]
        try:
            content = filepath.read_text(encoding="utf-8")
        except Exception:
            content = filepath.read_text(encoding="utf-8", errors="replace")

        if not content.strip():
            return []

        doc_id_input = f"{self.config.id}:{filepath.name}"
        doc_id = hashlib.sha256(doc_id_input.encode("utf-8")).hexdigest()[:16]
        title = _extract_markdown_title(content, filepath.stem)

        doc = CorpusDocument(
            id=doc_id,
            source_id=self.config.id,
            source_url=self.config.url or f"file://{filepath.as_posix()}",
            license_id=self.config.license_id,
            category=self.config.category,
            title=title,
            text=content,
            language=self.config.language,
            metadata={
                "file_name": filepath.name,
                "source_name": self.config.name,
                "license_source": "declared_manifest",
                "verified_license_id": self.config.license_id,
                "verified_commit_sha": "local",
                **self.config.metadata,
            },
        )
        return [doc]


class GitRepositorySourceAdapter(BaseSourceAdapter):
    """Clones remote Git repository into cache and ingests documentation with license check."""

    def __init__(self, config: SourceConfig, cache_dir: str | Path | None = None) -> None:
        super().__init__(config, cache_dir)
        self.repo_cache_dir = self.cache_dir / "git" / self.config.id

    def _clone_or_update_repo(self) -> Path:
        if not self.config.url:
            raise ValueError(f"Git source '{self.config.id}' requires 'url'")

        self.repo_cache_dir.parent.mkdir(parents=True, exist_ok=True)

        sparse_dirs = extract_sparse_checkout_dirs(self.config.include_patterns)

        if not (self.repo_cache_dir / ".git").exists():
            if sparse_dirs:
                cmd = [
                    "git",
                    "-c",
                    "core.longpaths=true",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    self.config.url,
                    str(self.repo_cache_dir),
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    raise RuntimeError(
                        f"Failed to clone git repo '{self.config.url}': {res.stderr.strip()}"
                    )
                sparse_cmd = ["git", "sparse-checkout", "set"] + sparse_dirs
                subprocess.run(sparse_cmd, cwd=self.repo_cache_dir, capture_output=True, text=True)
            else:
                cmd = [
                    "git",
                    "-c",
                    "core.longpaths=true",
                    "clone",
                    "--depth",
                    "1",
                    self.config.url,
                    str(self.repo_cache_dir),
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    raise RuntimeError(
                        f"Failed to clone git repo '{self.config.url}': {res.stderr.strip()}"
                    )
        elif not sparse_dirs:
            # A cache created for an earlier, narrower manifest may still be
            # sparse. Disable it before discovery to expose current patterns.
            sparse_status = subprocess.run(
                ["git", "sparse-checkout", "list"],
                cwd=self.repo_cache_dir,
                capture_output=True,
                text=True,
            )
            if sparse_status.returncode == 0:
                subprocess.run(
                    ["git", "sparse-checkout", "disable"],
                    cwd=self.repo_cache_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                )
        return self.repo_cache_dir

    def _get_commit_sha(self, repo_dir: Path) -> str:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True
        )
        if res.returncode == 0:
            return res.stdout.strip()
        return "unknown"

    def list_candidate_files(self) -> list[Path]:
        if not self.repo_cache_dir.exists():
            return []

        filepaths: list[Path] = []
        for p in self.repo_cache_dir.rglob("*"):
            if p.is_file() and matches_patterns(
                p, self.repo_cache_dir, self.config.include_patterns, self.config.exclude_patterns
            ):
                filepaths.append(p)
        return sorted(filepaths)

    def ingest(self) -> list[CorpusDocument]:
        repo_dir = self._clone_or_update_repo()
        commit_sha = self._get_commit_sha(repo_dir)

        # Strict repository-level license verification
        verification = verify_repository_license(repo_dir, self.config.license_id)
        if not verification.is_valid:
            logger.warning(
                "[LICENSE REJECTED] Source '%s' disabled: %s",
                self.config.id,
                verification.error_message,
            )
            return []

        filepaths = self.list_candidate_files()
        documents: list[CorpusDocument] = []

        base_url = self.config.url or ""
        if base_url.endswith(".git"):
            base_url = base_url[:-4]

        for filepath in filepaths:
            try:
                content = filepath.read_text(encoding="utf-8")
            except Exception:
                content = filepath.read_text(encoding="utf-8", errors="replace")

            if not content.strip():
                continue

            rel_path = filepath.relative_to(repo_dir).as_posix()
            doc_id_input = f"{self.config.id}:{commit_sha}:{rel_path}"
            doc_id = hashlib.sha256(doc_id_input.encode("utf-8")).hexdigest()[:16]

            if "github.com" in base_url:
                source_url = f"{base_url}/blob/{commit_sha}/{rel_path}"
            else:
                source_url = f"{base_url}#{rel_path}"

            title = _extract_markdown_title(content, filepath.stem)

            doc = CorpusDocument(
                id=doc_id,
                source_id=self.config.id,
                source_url=source_url,
                license_id=verification.verified_license_id,
                category=self.config.category,
                title=title,
                text=content,
                language=self.config.language,
                metadata={
                    "commit_sha": commit_sha,
                    "relative_path": rel_path,
                    "repository_url": self.config.url,
                    "license_source": verification.license_source,
                    "verified_license_id": verification.verified_license_id,
                    "verified_commit_sha": commit_sha,
                    "source_name": self.config.name,
                    **self.config.metadata,
                },
            )
            documents.append(doc)

        return documents


class HTMLBoilerplateCleaner(HTMLParser):
    """Extracts prose content, code blocks, and headings from HTML, stripping boilerplate."""

    IGNORED_TAGS = {
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "form",
        "noscript",
        "iframe",
        "button",
        "svg",
    }

    def __init__(self) -> None:
        super().__init__()
        self.output: list[str] = []
        self.tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        self.tag_stack.append(normalized)
        if normalized in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote"}:
            self.output.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.tag_stack and self.tag_stack[-1] == tag.lower():
            self.tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if any(t in self.IGNORED_TAGS for t in self.tag_stack):
            return
        cleaned = " ".join(data.split())
        if cleaned:
            tag = self.tag_stack[-1] if self.tag_stack else ""
            if tag.startswith("h") and len(tag) == 2 and tag[1].isdigit():
                self.output.append("#" * int(tag[1]) + " " + cleaned)
            elif tag == "li":
                self.output.append("- " + cleaned)
            elif tag in {"pre", "code"}:
                self.output.append("```\n" + cleaned + "\n```")
            else:
                self.output.append(cleaned)

    def get_text(self) -> str:
        return "\n".join(self.output)


class HttpDocumentationSourceAdapter(BaseSourceAdapter):
    """Ingests explicitly approved web documentation over HTTP/HTTPS."""

    def list_candidate_files(self) -> list[Path]:
        return []

    def ingest(self) -> list[CorpusDocument]:
        if not self.config.enabled:
            return []

        if not self.config.url:
            raise ValueError(f"HTTP source '{self.config.id}' requires 'url'")

        if not self.config.license_id:
            return []

        headers = {"User-Agent": "ArchitectAI-Corpus-Ingester/1.0"}
        response = requests.get(self.config.url, headers=headers, timeout=15)
        response.raise_for_status()

        html_content = response.text
        cleaner = HTMLBoilerplateCleaner()
        cleaner.feed(html_content)
        extracted_text = cleaner.get_text()

        if not extracted_text.strip():
            return []

        doc_id_input = f"{self.config.id}:{self.config.url}"
        doc_id = hashlib.sha256(doc_id_input.encode("utf-8")).hexdigest()[:16]

        doc = CorpusDocument(
            id=doc_id,
            source_id=self.config.id,
            source_url=self.config.url,
            license_id=self.config.license_id,
            category=self.config.category,
            title=self.config.name,
            text=extracted_text,
            language=self.config.language,
            metadata={
                "canonical_url": self.config.url,
                "source_name": self.config.name,
                "license_source": "declared_manifest",
                "verified_license_id": self.config.license_id,
                "verified_commit_sha": "http",
                **self.config.metadata,
            },
        )
        return [doc]


def _extract_markdown_title(content: str, default_stem: str) -> str:
    """Extract Markdown or plain-text AsciiDoc document title."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped.startswith("= "):
            return stripped[2:].strip()
    return default_stem.replace("_", " ").replace("-", " ").title()


def _resolve_configured_path(path: str) -> Path:
    """Resolve YAML paths including ``${ARCHITECT_DATA_DIR}`` on Windows."""
    expanded = os.path.expandvars(path)
    marker = "${ARCHITECT_DATA_DIR}"
    if marker in expanded:
        expanded = expanded.replace(marker, os.getenv("ARCHITECT_DATA_DIR", marker))
    result = Path(expanded)
    return result if result.is_absolute() else Path.cwd() / result


def _verify_local_source_license(
    root: Path, config: SourceConfig
) -> LicenseVerificationResult | None:
    if not config.verify_license:
        return None
    return verify_repository_license(root, config.license_id)


def _read_source_text(filepath: Path, parser: str | None, errors: str = "strict") -> str:
    """Read configured content deterministically; HTML extraction is prose-first."""
    raw = filepath.read_text(encoding="utf-8", errors=errors)
    use_html = (parser or "").lower() == "html" or filepath.suffix.lower() in {".html", ".htm"}
    if not use_html:
        return raw
    cleaner = HTMLBoilerplateCleaner()
    cleaner.feed(raw)
    return cleaner.get_text()


def _matches_allowed_content_type(path: Path, allowed: list[str] | None) -> bool:
    if not allowed:
        return True
    content_type = {
        ".md": "markdown",
        ".markdown": "markdown",
        ".adoc": "asciidoc",
        ".html": "html",
        ".htm": "html",
        ".txt": "text",
        ".rst": "restructuredtext",
    }.get(path.suffix.lower())
    return content_type in {item.lower() for item in allowed}


def _resolve_content_license(
    filepath: Path, base_dir: Path, config: SourceConfig
) -> dict[str, Any]:
    """Apply explicit content-class licensing rules; never infer a license."""
    policy = config.license_policy
    if policy.get("mode") != "path_scoped":
        return {
            "license_id": config.license_id,
            "training_enabled": True,
            "policy_type": "repository_wide",
            "evidence_path": config.license_evidence_path,
        }
    for rule in policy.get("rules", []):
        if matches_patterns(
            filepath,
            base_dir,
            rule.get("include_patterns") or [],
            rule.get("exclude_patterns") or [],
        ):
            return {
                "license_id": rule.get("license_id"),
                "training_enabled": bool(rule.get("training_enabled", True)),
                "policy_type": "path_scoped",
                "evidence_path": rule.get("evidence_path") or config.license_evidence_path,
            }
    return {
        "license_id": None,
        "training_enabled": False,
        "policy_type": "path_scoped",
        "evidence_path": config.license_evidence_path,
    }


def get_adapter(
    config: SourceConfig, cache_dir: str | Path | None = None
) -> BaseSourceAdapter:
    """Factory creating appropriate SourceAdapter instance for a SourceConfig."""
    adapter_type = config.type.lower()
    if adapter_type == "git_repository":
        return GitRepositorySourceAdapter(config, cache_dir)
    elif adapter_type == "markdown_directory":
        return MarkdownDirectorySourceAdapter(config, cache_dir)
    elif adapter_type == "local_directory":
        return LocalDirectorySourceAdapter(config, cache_dir)
    elif adapter_type == "local_file":
        return LocalFileSourceAdapter(config, cache_dir)
    elif adapter_type == "http_documentation":
        return HttpDocumentationSourceAdapter(config, cache_dir)
    else:
        raise ValueError(f"Unsupported source adapter type: '{config.type}'")


def load_source_manifest(config_path: str | Path) -> list[SourceConfig]:
    """Load and parse the sources.yaml manifest file.

    Returns:
        List of configured SourceConfig objects.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Source manifest not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "sources" not in data:
        return []

    configs: list[SourceConfig] = []
    for s_dict in data.get("sources", []):
        config = SourceConfig(
            id=s_dict["id"],
            name=s_dict.get("name", s_dict["id"]),
            category=s_dict["category"],
            enabled=s_dict.get("enabled", True),
            type=s_dict.get("type", "local_directory"),
            path=s_dict.get("path"),
            url=s_dict.get("url"),
            license_id=s_dict.get("license_id"),
            language=s_dict.get("language", "en"),
            include_patterns=s_dict.get("include_patterns"),
            exclude_patterns=s_dict.get("exclude_patterns"),
            allowed_content_types=s_dict.get("allowed_content_types"),
            parser=s_dict.get("parser"),
            source_token_cap=s_dict.get("source_token_cap"),
            source_priority=float(s_dict.get("source_priority", 1.0)),
            notes=s_dict.get("notes"),
            verify_license=bool(s_dict.get("verify_license", False)),
            allow_unverified_license=bool(s_dict.get("allow_unverified_license", False)),
            license_training_status=s_dict.get("license_training_status", "unverified"),
            release_eligible=bool(s_dict.get("release_eligible", False)),
            license_review_status=s_dict.get("license_review_status", "needs_manual_review"),
            license_evidence_path=s_dict.get("license_evidence_path"),
            license_policy=s_dict.get("license_policy", {}),
            strip_section_patterns=s_dict.get("strip_section_patterns"),
            section_category_rules=s_dict.get("section_category_rules", {}),
            metadata=s_dict.get("metadata", {}),
        )
        configs.append(config)

    return configs

