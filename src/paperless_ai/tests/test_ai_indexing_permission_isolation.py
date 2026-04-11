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
