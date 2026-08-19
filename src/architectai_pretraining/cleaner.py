"""Deterministic text cleaning pipeline for continuous pretraining text."""

import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Literal

from architectai_pretraining.models import CorpusDocument

UnicodeForm = Literal["NFC", "NFD", "NFKC", "NFKD"]


class BaseCleaner(ABC):
    """Abstract base class for format-specific text cleaners."""

    @abstractmethod
    def clean_text(self, text: str) -> str:
        """Clean and normalize input text string."""
        pass

    def clean_document(self, doc: CorpusDocument) -> CorpusDocument:
        """Apply text cleaning to a CorpusDocument and return a new updated document."""
        cleaned_text = self.clean_text(doc.text)
        return doc.model_copy(update={"text": cleaned_text})


class TextCleaner(BaseCleaner):
    """Deterministic general-purpose text cleaner.

    Normalizes unicode, line endings, whitespace, and strips unprintable control characters
    without rewriting or altering domain concepts.
    """

    def __init__(
        self,
        unicode_form: UnicodeForm = "NFC",
        max_consecutive_newlines: int = 2,
        strip_control_chars: bool = True,
    ):
        self.unicode_form: UnicodeForm = unicode_form
        self.max_consecutive_newlines = max_consecutive_newlines
        self.strip_control_chars = strip_control_chars

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        # 1. Unicode normalization
        text = unicodedata.normalize(self.unicode_form, text)

        # 2. Line ending normalization (CRLF / CR -> LF)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Strip null and unwanted control characters (keep tab \t and newline \n)
        if self.strip_control_chars:
            text = "".join(
                ch for ch in text if ch == "\n" or ch == "\t" or (ord(ch) >= 32 and ord(ch) != 127)
            )

        # 4. Strip trailing spaces/tabs on each line
        lines = [line.rstrip(" \t") for line in text.split("\n")]
        text = "\n".join(lines)

        # 5. Normalize excessive blank lines (e.g. max 2 consecutive newlines)
        if self.max_consecutive_newlines > 0:
            pattern = r"\n{" + str(self.max_consecutive_newlines + 1) + r",}"
            replacement = "\n" * self.max_consecutive_newlines
            text = re.sub(pattern, replacement, text)

        # 6. Normalize trailing/leading document whitespace
        return text.strip()


class MarkdownCleaner(TextCleaner):
    """Specialized cleaner for Markdown documentation and ADRs.

    Inherits standard text cleaning and can apply Markdown-specific normalization rules.
    """

    def clean_text(self, text: str) -> str:
        cleaned = super().clean_text(text)
        return cleaned


class BoilerplateCleaner:
    """Detects and strips repeated boilerplate blocks (KEP headers, footers, badges)."""

    def __init__(self) -> None:
        self.boilerplate_blocks_removed: int = 0
        self.characters_removed: int = 0
        self.documents_affected: int = 0

    def clean_document(self, doc: CorpusDocument) -> CorpusDocument:
        text = doc.text
        original_len = len(text)
        blocks_removed = 0

        # 1. Strip Markdown badges
        new_text, n1 = re.subn(r"\[!\[.*?\]\(.*?\)]\(.*?\)\s*", "", text)
        blocks_removed += n1

        # 2. Strip KEP YAML frontmatter headers if present at top
        new_text, n2 = re.subn(
            r"^---\s*\ntitle:[\s\S]*?\n---\s*\n", "", new_text, flags=re.MULTILINE
        )
        blocks_removed += n2

        # 3. Strip standard HTML comment boilerplate
        new_text, n3 = re.subn(r"<!--[\s\S]*?-->\s*", "", new_text)
        blocks_removed += n3

        new_text = new_text.strip()
        diff_len = original_len - len(new_text)

        if blocks_removed > 0 and diff_len > 0:
            self.boilerplate_blocks_removed += blocks_removed
            self.characters_removed += diff_len
            self.documents_affected += 1
            return doc.model_copy(update={"text": new_text})

        return doc
