"""
Unit tests for the widening/dedupe loop inside
`indexing.query_similar_documents`. These tests mock
`VectorIndexRetriever` so we can control exactly what each retrieve
pass returns, which is not possible against a real FAISS build.
"""

from unittest.mock import MagicMock, patch

import pytest
from django.utils import timezone

from documents.models import Document
from paperless_ai import indexing
from llama_index.core.schema import NodeWithScore, TextNode


def _mk_node(document_id: str, text: str = "x"):
    return NodeWithScore(
        node=TextNode(
            text=text,
            metadata={"document_id": document_id, "title": f"Doc {document_id}"},
        ),
        score=0.9,
    )


@pytest.fixture
def mocked_index():
    """Stub the load_or_build_index + vector_store_file_exists path."""
    with (
        patch(
            "paperless_ai.indexing.vector_store_file_exists",
            return_value=True,
        ),
        patch("paperless_ai.indexing.load_or_build_index") as mock_load,
    ):
        index = MagicMock()
        mock_load.return_value = index
        yield index


def _patch_retriever_with_passes(pass_returns):
    """Patch VectorIndexRetriever inside indexing so each instantiation
    consumes the next entry from pass_returns (a list of node lists)."""
    constructed = []

    def _factory(**kwargs):
        retriever = MagicMock()
        idx = len(constructed)
        if idx >= len(pass_returns):
            # Default: return the LAST pass's result (plateau behaviour)
            retriever.retrieve.return_value = pass_returns[-1]
        else:
            retriever.retrieve.return_value = pass_returns[idx]
        retriever._top_k = kwargs.get("similarity_top_k")
        constructed.append(retriever)
        return retriever

    return patch(
        "llama_index.core.retrievers.VectorIndexRetriever",
        side_effect=_factory,
    ), constructed


@pytest.mark.django_db
def test_query_similar_documents_deduplicates_chunks_from_same_doc(mocked_index):
    """Many chunks from the same document count as ONE distinct document.
    The loop must not exhaust its limit on repeated chunks."""
    query_doc = Document.objects.create(
        title="Query", content="x", added=timezone.now()
    )
    d1 = Document.objects.create(title="D1", content="x", added=timezone.now())
    d2 = Document.objects.create(title="D2", content="x", added=timezone.now())
    d3 = Document.objects.create(title="D3", content="x", added=timezone.now())
    d4 = Document.objects.create(title="D4", content="x", added=timezone.now())

    # Raw retrieve returns 10 chunks all from d1, then 3 from
    # other docs. Document-level dedupe must collapse the first 10
    # into a single entry for d1 and proceed to fill from d2, d3, d4.
    raw = (
        [_mk_node(str(d1.pk), f"chunk {i}") for i in range(10)]
        + [_mk_node(str(d2.pk)), _mk_node(str(d3.pk)), _mk_node(str(d4.pk))]
    )
    retriever_patch, constructed = _patch_retriever_with_passes([raw])
    with retriever_patch:
        results = indexing.query_similar_documents(
            document=query_doc, top_k=4, document_ids=None,
        )

    returned_pks = [d.pk for d in results]
    # d1 appears once, then d2, d3, d4 in rank order.
    assert returned_pks == [d1.pk, d2.pk, d3.pk, d4.pk]
    # Only one retrieve pass was needed.
    assert len(constructed) == 1


@pytest.mark.django_db
def test_query_similar_documents_widens_when_narrow_scope_pushes_matches_down(
    mocked_index,
):
    """Narrow document_ids: first pass top_k=100 returns only disallowed
    chunks; second pass (top_k=400) finally surfaces allowed ones."""
    query_doc = Document.objects.create(
        title="Query", content="x", added=timezone.now()
    )
    allowed = Document.objects.create(
        title="Allowed", content="x", added=timezone.now()
    )
    # Use a sentinel disallowed id that cannot collide with any real PK.
    disallowed_id = "99999999"

    # Pass 1 (top_k = max(5*20, 100) = 100): all disallowed
    pass1 = [_mk_node(disallowed_id) for _ in range(100)]
    # Pass 2 (top_k = 400): disallowed nodes plus the allowed one.
    # Only 1 allowed is found (< top_k=5) so the loop continues.
    pass2 = [_mk_node(disallowed_id) for _ in range(399)] + [_mk_node(str(allowed.pk))]
    # Pass 3 (top_k = 1000): same size as pass2 — plateau triggers exit.
    pass3 = [_mk_node(disallowed_id) for _ in range(399)] + [_mk_node(str(allowed.pk))]

    retriever_patch, constructed = _patch_retriever_with_passes([pass1, pass2, pass3])
    with retriever_patch:
        results = indexing.query_similar_documents(
            document=query_doc, top_k=5, document_ids=[allowed.pk],
        )

    returned_pks = [d.pk for d in results]
    assert returned_pks == [allowed.pk]
    # Three passes: widening kicked in twice; plateau on pass 3 stops it.
    assert len(constructed) == 3
    assert constructed[0]._top_k == 100
    assert constructed[1]._top_k == 400
    assert constructed[2]._top_k == 1000


@pytest.mark.django_db
def test_query_similar_documents_stops_on_raw_count_plateau(mocked_index):
    """When widening does not return more raw nodes, the loop stops
    early rather than spinning to RAW_MAX_TOP_K."""
    query_doc = Document.objects.create(
        title="Query", content="x", added=timezone.now()
    )
    # A real document that will NOT be in the allowed set (sentinel
    # allowed id below does not match this doc's pk).
    Document.objects.create(title="Irrelevant", content="x", added=timezone.now())

    # Both passes return the SAME raw count (plateau). No allowed matches —
    # use sentinel non-existent doc ids in the raw nodes.
    raw = [_mk_node("99999999") for _ in range(50)]
    retriever_patch, constructed = _patch_retriever_with_passes([raw, raw])
    with retriever_patch:
        results = indexing.query_similar_documents(
            document=query_doc,
            top_k=5,
            document_ids=[88888888],  # sentinel — matches nothing in raw
        )

    assert results == []
    # Exactly two passes — the second plateau pass triggers the exit.
    assert len(constructed) == 2


@pytest.mark.django_db
def test_query_similar_documents_hard_stops_at_raw_max_top_k(mocked_index):
    """When every widened pass returns strictly more raw nodes and no
    allowed matches, the loop terminates at RAW_MAX_TOP_K."""
    query_doc = Document.objects.create(
        title="Query", content="x", added=timezone.now()
    )

    # top_k progression for top_k=5: 100 → 400 → 1000 (RAW_MAX_TOP_K = max(5*100, 1000)).
    # Each pass returns strictly more raw nodes so plateau never fires.
    pass1 = [_mk_node("99999999") for _ in range(100)]
    pass2 = [_mk_node("99999999") for _ in range(400)]
    pass3 = [_mk_node("99999999") for _ in range(1000)]

    retriever_patch, constructed = _patch_retriever_with_passes([pass1, pass2, pass3])
    with retriever_patch:
        results = indexing.query_similar_documents(
            document=query_doc, top_k=5, document_ids=[88888888],
        )

    assert results == []
    top_ks = [r._top_k for r in constructed]
    assert top_ks == [100, 400, 1000]


@pytest.mark.django_db
def test_query_similar_documents_returns_partial_on_mid_loop_exception(
    mocked_index,
):
    """Pass 1 returns 2 allowed docs (not enough for top_k=5), pass 2
    raises. The function catches, breaks out of the loop, and returns
    the 2 docs it already had."""
    query_doc = Document.objects.create(
        title="Query", content="x", added=timezone.now()
    )
    d1 = Document.objects.create(title="D1", content="x", added=timezone.now())
    d2 = Document.objects.create(title="D2", content="x", added=timezone.now())

    # Two allowed, but top_k=5 asks for more — forces a second pass.
    pass1 = [_mk_node(str(d1.pk)), _mk_node(str(d2.pk))]

    # Custom patch that raises on the second retrieve.
    constructed = []

    def _factory(**kwargs):
        idx = len(constructed)
        retriever = MagicMock()
        retriever._top_k = kwargs.get("similarity_top_k")
        if idx == 0:
            retriever.retrieve.return_value = pass1
        else:
            retriever.retrieve.side_effect = RuntimeError("faiss boom")
        constructed.append(retriever)
        return retriever

    with patch(
        "llama_index.core.retrievers.VectorIndexRetriever",
        side_effect=_factory,
    ):
        results = indexing.query_similar_documents(
            document=query_doc, top_k=5, document_ids=None,
        )

    # Partial result preserved: pass 1's 2 allowed docs are returned
    # because pass 1 had written into seen_document_ids before pass 2
    # raised and broke the loop.
    returned_pks = {d.pk for d in results}
    assert returned_pks == {d1.pk, d2.pk}
    assert len(constructed) == 2
