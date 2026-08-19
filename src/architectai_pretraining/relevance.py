"""Domain relevance gate for filtering non-architecture documentation."""

import re
from dataclasses import dataclass

from architectai_pretraining.models import CorpusDocument


@dataclass
class RelevanceCheckResult:
    is_relevant: bool
    reason: str | None = None
    category: str | None = None


NON_RELEVANT_PATH_PATTERNS = [
    r"contributing\.md$",
    r"code_of_conduct\.md$",
    r"changelog\.md$",
    r"releasenotes\.",
    r"issue_template",
    r"pull_request_template",
    r"governance\.md$",
    r"citation\.cff$",
    r"template\.md$",
]

NON_RELEVANT_TITLE_PATTERNS = [
    r"^contributing to",
    r"^code of conduct",
    r"^release notes",
    r"^changelog",
    r"^how to contribute",
    r"^governance",
    r"^bug report",
    r"^feature request",
]


class DomainRelevanceGate:
    """Deterministic domain relevance filter for software architecture corpus."""

    def check(self, doc: CorpusDocument) -> RelevanceCheckResult:
        doc_id = doc.id.lower()
        title_lower = (doc.title or "").lower()
        text = doc.text

        # 1. Path & Filename pattern check
        for pat in NON_RELEVANT_PATH_PATTERNS:
            if re.search(pat, doc_id):
                return RelevanceCheckResult(
                    is_relevant=False,
                    reason=f"Path matched non-architecture metadata pattern: {pat}",
                    category="project_metadata",
                )

        # 2. Title pattern check
        for pat in NON_RELEVANT_TITLE_PATTERNS:
            if re.search(pat, title_lower):
                return RelevanceCheckResult(
                    is_relevant=False,
                    reason=f"Title matched non-architecture pattern: {pat}",
                    category="project_metadata",
                )

        # 3. Generated Index / Link dump check
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 5:
            link_lines = sum(1 for line in lines if line.startswith(("- [", "* [", "1. [")))
            if link_lines / len(lines) > 0.85 and len(lines) > 10:
                return RelevanceCheckResult(
                    is_relevant=False,
                    reason="Document is a generated link/navigation index dump.",
                    category="generated_index",
                )

        # 4. Pure installation script check
        if title_lower.startswith("installation") or "how to install" in title_lower:
            words = text.split()
            if len(words) < 200 and ("apt-get" in text or "brew install" in text or "docker run" in text):
                return RelevanceCheckResult(
                    is_relevant=False,
                    reason="Pure installation/command reference without architecture explanation.",
                    category="installation_only",
                )

        return RelevanceCheckResult(is_relevant=True, reason=None, category=None)
