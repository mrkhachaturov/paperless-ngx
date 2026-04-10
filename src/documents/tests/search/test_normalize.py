from __future__ import annotations

import pytest

from documents.search._normalize import ascii_fold

pytestmark = pytest.mark.search


class TestAsciiFold:
    """Unit tests for the _normalize.ascii_fold helper.

    Covers autocomplete word normalization (the only direct caller
    of ascii_fold in the search subsystem), pinned to the contract
    that the helper must strip Latin diacritics for search consistency
    while preserving non-Latin scripts byte-for-byte.
    """

    def test_preserves_cyrillic_lowercase(self) -> None:
        """Cyrillic lowercase passes through unchanged (no combining marks)."""
        assert ascii_fold("водоканал") == "водоканал"

    def test_preserves_cyrillic_titlecase(self) -> None:
        """Cyrillic capital letters pass through unchanged — no mark stripping."""
        assert ascii_fold("Водоканал") == "Водоканал"

    def test_folds_latin_diacritics(self) -> None:
        """NFD decomposition + combining-mark strip yields plain ASCII for café."""
        assert ascii_fold("café") == "cafe"
        assert ascii_fold("naïve") == "naive"
        assert ascii_fold("résumé") == "resume"

    def test_preserves_precomposed_latin_ligatures(self) -> None:
        """Precomposed ligatures have no combining marks, so NFD leaves them alone.

        Documented divergence from Tantivy's Rust-side fold, accepted
        for our Russian-only scope — see the design doc for details.
        """
        assert ascii_fold("æther") == "æther"
        assert ascii_fold("Straße") == "Straße"
