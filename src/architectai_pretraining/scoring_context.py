"""Structural context used only for deterministic corpus scoring."""

from architectai_pretraining.models import CorpusDocument


def contextual_scoring_view(doc: CorpusDocument) -> str:
    """Return deterministic structural context without changing training text.

    Derived paragraph chunks intentionally retain only their original prose in
    ``doc.text``.  Titles and headings are supplied here solely to scoring and
    semantic decisions, so a later chunk is not structurally anonymous.
    """
    parts: list[str] = []
    for value in (doc.title, doc.section_title, *doc.section_headings):
        normalized = " ".join((value or "").split())
        if normalized and f"# {normalized}" not in parts:
            # Marker exists only in the ephemeral scoring view; it preserves
            # structural credit without injecting synthetic training prose.
            parts.append(f"# {normalized}")
    parts.append(doc.text)
    return "\n".join(parts)
