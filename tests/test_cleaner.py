"""Tests for deterministic text cleaner."""

from architectai_pretraining.cleaner import TextCleaner
from architectai_pretraining.models import CorpusDocument


def test_text_cleaner_crlf_and_whitespace_normalization() -> None:
    cleaner = TextCleaner()
    raw_text = "Line 1\r\nLine 2   \r\n\r\n\r\n\r\nLine 3 \t "
    cleaned = cleaner.clean_text(raw_text)

    assert "\r" not in cleaned
    assert "Line 1\nLine 2" in cleaned
    # Excessive blank lines reduced to 2
    assert "\n\n\n" not in cleaned
    assert cleaned.endswith("Line 3")


def test_text_cleaner_unicode_normalization() -> None:
    cleaner = TextCleaner(unicode_form="NFC")
    # Decomposed e + acute accent -> NFC composed e with acute accent
    decomposed = "e\u0301tude"
    cleaned = cleaner.clean_text(decomposed)

    assert cleaned == "\u00e9tude"


def test_text_cleaner_strips_null_and_control_chars() -> None:
    cleaner = TextCleaner()
    raw_text = "Hello\x00 World!\x07\x0bKeep\tTab and\nNewline"
    cleaned = cleaner.clean_text(raw_text)

    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    assert "Hello World!Keep\tTab and\nNewline" == cleaned


def test_clean_document_updates_text() -> None:
    cleaner = TextCleaner()
    doc = CorpusDocument(
        id="doc-clean-1",
        source_id="src-1",
        category="reliability",
        text="Raw text with trailing space   \r\n\r\n\r\nand CRLF.",
    )
    cleaned_doc = cleaner.clean_document(doc)

    assert cleaned_doc.id == doc.id
    assert cleaned_doc.text == "Raw text with trailing space\n\nand CRLF."
