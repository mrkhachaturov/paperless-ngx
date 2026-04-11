"""
Permission isolation regression test for query_similar_documents.

This test exercises the real LlamaIndex + FAISS code path — no mocks
on the retriever or the vector store. It is deliberately authored to
prove the current bug: passing `document_ids` as a scoping filter
returns cross-group data because the upstream code's docstore-level
filter has a type mismatch (str metadata vs int parameter) AND passes
chunk node IDs as `doc_ids` (which expects ref_doc_id), and the
vendored FAISS integration does not filter at query time.

After Chain C's Patch C.2 lands, this test goes green because the
rewritten query_similar_documents uses a Python post-filter on
`node.metadata['document_id']` with normalized IDs.

Setup discipline: top_k is set to 50 (>> total node count), so
ranking from `FakeEmbedding` (identical vectors) cannot hide a leak.
Every node in the index is eligible to come back, and only the
permission filter can keep group B out.
"""

from unittest.mock import patch

import pytest
from django.utils import timezone

from documents.models import Document
from paperless_ai import indexing


@pytest.mark.django_db
def test_query_similar_documents_isolates_by_document_ids(
    temp_llm_index_dir,
    mock_embed_model,
):
    """
    Two disjoint groups of 5 documents each. A query scoped to group A
    must never return group B documents, regardless of ranking.
    """
    group_a_docs = [
        Document.objects.create(
            title=f"Alpha {i}",
            content=f"alpha content number {i}",
            added=timezone.now(),
        )
        for i in range(5)
    ]
    group_b_docs = [
        Document.objects.create(
            title=f"Beta {i}",
            content=f"beta content number {i}",
            added=timezone.now(),
        )
        for i in range(5)
    ]

    indexing.update_llm_index(rebuild=True)

    # top_k >> total nodes in the index means ranking cannot hide a
    # leak. Every indexed node is eligible to come back.
    results = indexing.query_similar_documents(
        document=group_a_docs[0],
        top_k=50,
        document_ids=[d.pk for d in group_a_docs],  # ints, as production does
    )

    returned_pks = {d.pk for d in results}
    a_pks = {d.pk for d in group_a_docs}
    b_pks = {d.pk for d in group_b_docs}

    assert returned_pks.issubset(a_pks), (
        f"Permission leak in query_similar_documents: "
        f"results contain group B documents {returned_pks & b_pks}"
    )
    assert returned_pks, (
        "query_similar_documents returned zero documents even with "
        "top_k >> index size and document_ids covering the query doc's group"
    )


@pytest.mark.django_db
def test_query_similar_documents_empty_document_ids_is_unscoped(
    temp_llm_index_dir,
    mock_embed_model,
):
    """document_ids=[] means 'no filter' — same behaviour as None.
    This matches the upstream contract (if document_ids branch)."""
    docs = [
        Document.objects.create(
            title=f"Doc {i}",
            content=f"content {i}",
            added=timezone.now(),
        )
        for i in range(3)
    ]
    indexing.update_llm_index(rebuild=True)

    with_empty = indexing.query_similar_documents(
        document=docs[0], top_k=10, document_ids=[],
    )
    with_none = indexing.query_similar_documents(
        document=docs[0], top_k=10, document_ids=None,
    )
    assert {d.pk for d in with_empty} == {d.pk for d in with_none}


@pytest.mark.django_db
def test_query_similar_documents_none_document_ids_is_historical_default(
    temp_llm_index_dir,
    mock_embed_model,
):
    docs = [
        Document.objects.create(
            title=f"Doc {i}",
            content=f"content {i}",
            added=timezone.now(),
        )
        for i in range(3)
    ]
    indexing.update_llm_index(rebuild=True)

    results = indexing.query_similar_documents(
        document=docs[0], top_k=10, document_ids=None,
    )
    returned = {d.pk for d in results}
    # Historical default returns distinct documents up to top_k,
    # excluding the query document itself is optional — some
    # implementations include it, some don't. Assert we got at
    # least some results.
    assert len(returned) >= 1


@pytest.mark.django_db
def test_query_similar_documents_nonexistent_document_ids_returns_empty(
    temp_llm_index_dir,
    mock_embed_model,
):
    docs = [
        Document.objects.create(
            title=f"Doc {i}",
            content=f"content {i}",
            added=timezone.now(),
        )
        for i in range(3)
    ]
    indexing.update_llm_index(rebuild=True)

    results = indexing.query_similar_documents(
        document=docs[0],
        top_k=10,
        document_ids=[99998, 99999],  # no such documents
    )
    assert results == []


@pytest.mark.django_db
def test_query_similar_documents_mixed_int_str_document_ids_normalized(
    temp_llm_index_dir,
    mock_embed_model,
):
    """Normalization to set[str] at function entry must accept mixed
    int/str input without raising and without dropping either type."""
    docs = [
        Document.objects.create(
            title=f"Doc {i}",
            content=f"content {i}",
            added=timezone.now(),
        )
        for i in range(5)
    ]
    indexing.update_llm_index(rebuild=True)

    # Mix of int and str IDs
    mixed = [docs[0].pk, str(docs[1].pk), docs[2].pk]
    results = indexing.query_similar_documents(
        document=docs[0], top_k=10, document_ids=mixed,
    )
    returned = {d.pk for d in results}
    expected = {docs[0].pk, docs[1].pk, docs[2].pk}
    assert returned.issubset(expected)


@pytest.mark.django_db
def test_query_similar_documents_document_ids_covering_all_matches_unscoped(
    temp_llm_index_dir,
    mock_embed_model,
):
    """When document_ids covers the entire index, results should match
    the unscoped call for the same query."""
    docs = [
        Document.objects.create(
            title=f"Doc {i}",
            content=f"content {i}",
            added=timezone.now(),
        )
        for i in range(4)
    ]
    indexing.update_llm_index(rebuild=True)

    all_ids = [d.pk for d in docs]
    with_all = indexing.query_similar_documents(
        document=docs[0], top_k=10, document_ids=all_ids,
    )
    unscoped = indexing.query_similar_documents(
        document=docs[0], top_k=10, document_ids=None,
    )
    assert {d.pk for d in with_all} == {d.pk for d in unscoped}


@pytest.mark.django_db
def test_query_similar_documents_load_failure_returns_empty(
    temp_llm_index_dir,
    mock_embed_model,
    caplog,
):
    """load_or_build_index raising ValueError is caught; helper returns []."""
    import logging

    docs = [
        Document.objects.create(
            title="Solo",
            content="content",
            added=timezone.now(),
        ),
    ]

    with patch(
        "paperless_ai.indexing.load_or_build_index",
        side_effect=ValueError("index missing"),
    ), patch(
        "paperless_ai.indexing.vector_store_file_exists",
        return_value=True,
    ), caplog.at_level(logging.WARNING, logger="paperless_ai.indexing"):
        results = indexing.query_similar_documents(
            document=docs[0], top_k=5, document_ids=None,
        )

    assert results == []
    assert any("failed to load LLM index" in r.message for r in caplog.records)
