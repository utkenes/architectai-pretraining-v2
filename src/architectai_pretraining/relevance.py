"""Domain relevance gate for filtering non-architecture documentation."""

import re
from dataclasses import dataclass

from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.scoring_context import contextual_scoring_view


@dataclass
class RelevanceCheckResult:
    is_relevant: bool
    reason: str | None = None
    category: str | None = None


@dataclass
class ArchitectureRelevanceScore:
    score: float
    passed: bool
    reasons: list[str]
    link_ratio: float
    prose_density: float


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
    r"^contributors?$",
    r"^social links?$",
    r"^community$",
]


class DomainRelevanceGate:
    """Deterministic domain relevance filter for software architecture corpus."""

    def check(self, doc: CorpusDocument) -> RelevanceCheckResult:
        doc_id = (doc.relative_path or doc.id).lower()
        title_lower = " ".join([doc.title or "", doc.section_title or "", *doc.section_headings]).lower()
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


class ArchitectureRelevanceScorer:
    """Score explanatory architecture reasoning without relying on keywords alone."""

    SIGNALS = {
        "trade-off", "tradeoff", "alternative", "decision", "consequence", "constraint",
        "scalability", "consistency", "availability", "latency", "throughput", "failure",
        "replication", "partition", "transaction", "idempotency", "retry", "timeout",
        "resilience", "coupling", "cohesion", "boundary", "module", "service", "deployment",
        "data model", "storage", "queue", "messaging", "reliability", "maintainability",
        "migration", "rationale", "fault tolerance", "event-driven",
    }

    def __init__(self, min_score: float = 0.40, max_link_ratio: float = 0.22) -> None:
        self.min_score = min_score
        self.max_link_ratio = max_link_ratio

    def score(self, doc: CorpusDocument) -> ArchitectureRelevanceScore:
        scoring_text = contextual_scoring_view(doc)
        text = scoring_text.lower()
        words = re.findall(r"\b[a-z][a-z-]*\b", text)
        lines = [line.strip() for line in scoring_text.splitlines() if line.strip()]
        content_lines = [line.strip() for line in doc.text.splitlines() if line.strip()]
        word_count = max(1, len(words))
        signal_hits = sum(text.count(signal) for signal in self.SIGNALS)
        signal_score = min(1.0, signal_hits / max(2.0, word_count / 85.0))
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", doc.text))
        prose_density = min(1.0, sentence_count / max(1.0, len(lines) * 0.35))
        heading_count = sum(1 for line in lines if re.match(r"^(#{1,6}\s|={1,6}\s)", line))
        structure_score = min(1.0, heading_count / 3.0)
        adr_labels = sum(
            1 for label in ("context", "decision", "consequences", "alternatives", "status", "rationale")
            if re.search(rf"(?im)^#{1,6}\s+{label}\b|^{label}\s*$", scoring_text)
        )
        adr_score = min(1.0, adr_labels / 3.0)
        links = len(re.findall(r"https?://|\[[^\]]+\]\([^)]*\)", doc.text))
        link_ratio = links / max(1, len(content_lines))
        code_lines = sum(1 for line in content_lines if line.startswith(("```", "    ", "\t")))
        code_penalty = min(1.0, code_lines / max(1, len(content_lines)))
        # DomainRelevanceGate owns real navigation/link-dump rejection. Link
        # ratio remains an auditable soft signal for otherwise substantive
        # documentation, including sections above the historical ratio cap.
        link_penalty = max(0.0, (link_ratio / max(self.max_link_ratio, 0.001) - 0.5) * 2)
        score = 0.42 * signal_score + 0.28 * prose_density + 0.18 * structure_score + 0.12 * adr_score
        # CodeProseAnalyzer remains the authoritative code-dominance gate.
        score -= 0.06 * link_penalty + 0.05 * code_penalty
        score = round(max(0.0, min(1.0, score)), 4)
        reasons: list[str] = []
        if signal_hits:
            reasons.append(f"{signal_hits} architecture-reasoning signals")
        if adr_labels >= 2:
            reasons.append("ADR decision structure")
        if link_ratio > self.max_link_ratio:
            reasons.append("link-heavy reference material")
        if code_penalty > 0.5:
            reasons.append("code-heavy material")
        return ArchitectureRelevanceScore(
            score=score,
            passed=score >= self.min_score,
            reasons=reasons,
            link_ratio=round(link_ratio, 4),
            prose_density=round(prose_density, 4),
        )
