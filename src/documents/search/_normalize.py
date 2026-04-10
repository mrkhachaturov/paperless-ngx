from __future__ import annotations

import unicodedata


def ascii_fold(text: str) -> str:
    """
    Normalize unicode text for search by stripping combining diacritic marks.

    Uses NFD decomposition to separate base characters from their combining
    marks (e.g. 'é' → 'e' + U+0301), then drops the marks. Leaves non-Latin
    scripts (Cyrillic, Greek, CJK, Arabic, etc.) and precomposed ligatures
    ('æ', 'ß', 'œ') unchanged — a known divergence from Tantivy's Rust-side
    fold filter that is accepted for our Russian-only deployment.
    """
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")
