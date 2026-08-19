"""Quality gate rules and filtering for candidate pretraining documents."""

import re
from dataclasses import dataclass, field

from architectai_pretraining.models import CorpusDocument


@dataclass
class QualityGateConfig:
    """Configurable quality gate thresholds."""

    min_char_length: int = 80
    min_word_count: int = 15
    allowed_languages: set[str] = field(default_factory=lambda: {"en"})
    max_symbol_ratio: float = 0.40  # Max non-alphanumeric/non-space character ratio
    require_provenance: bool = True


@dataclass
class ValidationResult:
    """Result of passing a CorpusDocument through the quality gate."""

    passed: bool
    rejection_reasons: list[str]


class QualityGate:
    """Evaluates CorpusDocument objects against configured quality standards."""

    def __init__(self, config: QualityGateConfig | None = None):
        self.config = config or QualityGateConfig()

    def validate(self, doc: CorpusDocument) -> ValidationResult:
        reasons: list[str] = []
        text = doc.text.strip()
        char_count = len(text)
        words = text.split()
        word_count = len(words)

        # 1. Minimum character length check
        if char_count < self.config.min_char_length:
            reasons.append(
                f"Character count ({char_count}) below minimum threshold "
                f"({self.config.min_char_length})."
            )

        # 2. Minimum word count check
        if word_count < self.config.min_word_count:
            reasons.append(
                f"Word count ({word_count}) below minimum threshold "
                f"({self.config.min_word_count})."
            )

        # 3. Language check
        if doc.language not in self.config.allowed_languages:
            reasons.append(
                f"Language '{doc.language}' not in allowed set: {self.config.allowed_languages}."
            )

        # 4. Provenance check
        if self.config.require_provenance and not doc.source_id.strip():
            reasons.append("Missing required source provenance identifier.")

        # 5. Non-symbol / non-punctuation ratio check
        if char_count > 0:
            # Count alphanumeric characters vs total non-space characters
            alphanumeric_count = len(re.findall(r"\w", text))
            non_space_count = len(re.findall(r"\S", text))
            if non_space_count > 0:
                symbol_count = non_space_count - alphanumeric_count
                symbol_ratio = symbol_count / non_space_count
                if symbol_ratio > self.config.max_symbol_ratio:
                    reasons.append(
                        f"Symbol ratio ({symbol_ratio:.2f}) exceeds maximum threshold "
                        f"({self.config.max_symbol_ratio:.2f})."
                    )

        return ValidationResult(passed=len(reasons) == 0, rejection_reasons=reasons)

    def filter_documents(
        self, documents: list[CorpusDocument]
    ) -> tuple[list[CorpusDocument], list[tuple[CorpusDocument, list[str]]]]:
        """Filter documents through the quality gate.

        Returns:
            Tuple of (accepted_documents, list of (rejected_document, reasons))
        """
        accepted: list[CorpusDocument] = []
        rejected: list[tuple[CorpusDocument, list[str]]] = []

        for doc in documents:
            res = self.validate(doc)
            if res.passed:
                accepted.append(doc)
            else:
                rejected.append((doc, res.rejection_reasons))

        return accepted, rejected
