"""
Unit tests for paperless_ai.util helpers.

These tests deliberately avoid Django and real LlamaIndex integration.
They use MagicMock to stand in for BaseNode and NodeWithScore so the
helper can be tested for its own contract without pulling in the
framework.
"""

from unittest.mock import MagicMock

import pytest


def _fake_bare_node(document_id: str, text: str = "x"):
    """Return a MagicMock that duck-types as a BaseNode."""
    node = MagicMock()
    node.metadata = {"document_id": document_id, "title": f"doc-{document_id}"}
    node.text = text
    # BaseNode has no `.node` attribute — important for the helper's
    # duck-typing branch that distinguishes BaseNode from NodeWithScore.
    del node.node
    return node


def _fake_scored_node(document_id: str, text: str = "x", score: float = 0.5):
    """Return a MagicMock that duck-types as a NodeWithScore."""
    inner = MagicMock()
    inner.metadata = {"document_id": document_id, "title": f"doc-{document_id}"}
    inner.text = text
    scored = MagicMock()
    scored.node = inner
    scored.score = score
    # NodeWithScore also exposes metadata as a pass-through property per
    # the real schema, so tests that touch n.metadata directly should
    # also work. We mirror that here.
    scored.metadata = inner.metadata
    scored.text = inner.text
    return scored


def test_filter_nodes_by_doc_ids_bare_nodes_happy_path():
    from paperless_ai.util import filter_nodes_by_doc_ids

    nodes = [
        _fake_bare_node("1"),
        _fake_bare_node("2"),
        _fake_bare_node("3"),
    ]
    result = filter_nodes_by_doc_ids(nodes, doc_ids=["1", "3"], limit=10)
    assert [n.metadata["document_id"] for n in result] == ["1", "3"]


def test_filter_nodes_by_doc_ids_scored_nodes_duck_typed_through_node_attr():
    """NodeWithScore wrappers are filtered by inner node metadata but
    returned as wrappers — not unwrapped."""
    from paperless_ai.util import filter_nodes_by_doc_ids

    nodes = [
        _fake_scored_node("1", score=0.9),
        _fake_scored_node("2", score=0.8),
        _fake_scored_node("3", score=0.7),
    ]
    result = filter_nodes_by_doc_ids(nodes, doc_ids=["2"], limit=10)
    assert len(result) == 1
    # Must still be the scored wrapper, not the inner node
    assert result[0].score == 0.8
    assert result[0].node.metadata["document_id"] == "2"


def test_filter_nodes_by_doc_ids_preserves_input_order():
    from paperless_ai.util import filter_nodes_by_doc_ids

    nodes = [
        _fake_bare_node("3"),
        _fake_bare_node("1"),
        _fake_bare_node("2"),
    ]
    result = filter_nodes_by_doc_ids(nodes, doc_ids=["1", "2", "3"], limit=10)
    assert [n.metadata["document_id"] for n in result] == ["3", "1", "2"]


def test_filter_nodes_by_doc_ids_truncates_at_limit():
    from paperless_ai.util import filter_nodes_by_doc_ids

    nodes = [_fake_bare_node(str(i)) for i in range(10)]
    result = filter_nodes_by_doc_ids(
        nodes, doc_ids=[str(i) for i in range(10)], limit=3,
    )
    assert len(result) == 3
    assert [n.metadata["document_id"] for n in result] == ["0", "1", "2"]


def test_filter_nodes_by_doc_ids_empty_allowed_returns_empty():
    from paperless_ai.util import filter_nodes_by_doc_ids

    nodes = [_fake_bare_node("1"), _fake_bare_node("2")]
    result = filter_nodes_by_doc_ids(nodes, doc_ids=[], limit=10)
    assert result == []


def test_filter_nodes_by_doc_ids_no_matches_returns_empty():
    from paperless_ai.util import filter_nodes_by_doc_ids

    nodes = [_fake_bare_node("1"), _fake_bare_node("2")]
    result = filter_nodes_by_doc_ids(nodes, doc_ids=["999"], limit=10)
    assert result == []


def test_filter_nodes_by_doc_ids_missing_document_id_key_is_skipped():
    from paperless_ai.util import filter_nodes_by_doc_ids

    node_without_doc_id = MagicMock()
    node_without_doc_id.metadata = {"title": "orphan"}
    del node_without_doc_id.node

    nodes = [node_without_doc_id, _fake_bare_node("1")]
    result = filter_nodes_by_doc_ids(nodes, doc_ids=["1"], limit=10)
    assert len(result) == 1
    assert result[0].metadata["document_id"] == "1"


def _fake_global_index_with_retriever(nodes_by_top_k):
    """
    Build a MagicMock global_index whose `as_retriever(similarity_top_k=K)`
    returns a retriever whose `.retrieve(query_str)` returns
    nodes_by_top_k[K]. This lets tests assert exactly which
    similarity_top_k values the widening loop asked for.
    """
    index = MagicMock()
    calls = []

    def as_retriever(similarity_top_k):
        calls.append(similarity_top_k)
        retriever = MagicMock()
        retriever.retrieve = MagicMock(return_value=nodes_by_top_k[similarity_top_k])
        return retriever

    index.as_retriever = MagicMock(side_effect=as_retriever)
    # Attach the call log so tests can introspect it.
    index._as_retriever_top_k_calls = calls
    return index


def test_retrieve_with_attrition_budget_single_pass_common_case():
    """First pass returns enough allowed nodes; only one retriever is built."""
    from paperless_ai.util import retrieve_with_attrition_budget

    allowed = [_fake_scored_node(str(i)) for i in range(5)]
    # 25 raw nodes returned on the first pass, all allowed
    index = _fake_global_index_with_retriever({25: allowed})

    result = retrieve_with_attrition_budget(
        global_index=index,
        query_str="anything",
        allowed_doc_ids=[str(i) for i in range(5)],
        final_limit=5,
    )
    assert len(result.kept) == 5
    assert result.passes == 1
    assert result.final_top_k == 25
    assert result.raw_returned == 5
    assert index._as_retriever_top_k_calls == [25]


def test_retrieve_with_attrition_budget_widens_exponentially():
    """Three passes: 25 → 100 → 400. Allowed matches arrive only on the third."""
    from paperless_ai.util import retrieve_with_attrition_budget

    # First two passes return only disallowed docs, third returns enough
    # allowed ones. Use distinct sizes so the exhaustion detector does
    # NOT kick in before we reach the allowed pass.
    nodes_by_top_k = {
        25: [_fake_scored_node("X") for _ in range(25)],
        100: [_fake_scored_node("X") for _ in range(100)],
        400: (
            [_fake_scored_node("X") for _ in range(395)]
            + [_fake_scored_node("1"), _fake_scored_node("2"),
               _fake_scored_node("3"), _fake_scored_node("4"),
               _fake_scored_node("5")]
        ),
    }
    index = _fake_global_index_with_retriever(nodes_by_top_k)

    result = retrieve_with_attrition_budget(
        global_index=index,
        query_str="narrow scope",
        allowed_doc_ids=["1", "2", "3", "4", "5"],
        final_limit=5,
    )
    assert len(result.kept) == 5
    assert result.passes == 3
    assert result.final_top_k == 400
    assert index._as_retriever_top_k_calls == [25, 100, 400]


def test_retrieve_with_attrition_budget_hard_stops_at_max_top_k():
    """No allowed matches anywhere, each widened pass returns MORE raw
    nodes than the previous one, so the exhaustion detector never fires
    and the hard stop at max_top_k is the only exit."""
    from paperless_ai.util import retrieve_with_attrition_budget

    # 25 → 100 → 400 → 500 (capped). Each pass must return strictly more
    # nodes so prev_raw_count comparison never triggers early stop.
    nodes_by_top_k = {
        25:  [_fake_scored_node("X") for _ in range(25)],
        100: [_fake_scored_node("X") for _ in range(100)],
        400: [_fake_scored_node("X") for _ in range(400)],
        500: [_fake_scored_node("X") for _ in range(500)],
    }
    index = _fake_global_index_with_retriever(nodes_by_top_k)

    result = retrieve_with_attrition_budget(
        global_index=index,
        query_str="very narrow",
        allowed_doc_ids=["ALLOWED_NEVER_APPEARS"],
        final_limit=5,
    )
    assert result.kept == []
    assert result.passes == 4
    assert result.final_top_k == 500
    # Widening: 25 → 100 → 400 → 500, then max_top_k gate stops the loop.
    assert index._as_retriever_top_k_calls == [25, 100, 400, 500]


def test_retrieve_with_attrition_budget_stops_on_raw_count_plateau():
    """First pass returns 37 raw nodes, second pass returns the same 37
    (FAISS has nothing more to give). The loop must terminate without
    spinning up to max_top_k."""
    from paperless_ai.util import retrieve_with_attrition_budget

    the_37 = [_fake_scored_node("X") for _ in range(37)]
    nodes_by_top_k = {
        25: the_37[:25],  # 25 raw
        100: the_37,       # still 37 raw, fewer than top_k requested
        # If the loop widened further, 400 would fail this dict lookup
        # and the test would crash — that is the assertion.
    }
    index = _fake_global_index_with_retriever(nodes_by_top_k)

    result = retrieve_with_attrition_budget(
        global_index=index,
        query_str="small index",
        allowed_doc_ids=["NEVER_MATCHES"],
        final_limit=5,
    )
    # We exit because widening produced no additional raw nodes, not
    # because max_top_k was hit.
    assert result.kept == []
    assert result.passes == 2
    # Two passes at 25 and 100; no third.
    assert index._as_retriever_top_k_calls == [25, 100]


def test_retrieve_with_attrition_budget_catches_retriever_exceptions(caplog):
    """Retriever raises on the first call. Helper returns empty and
    logs via logger.exception — no propagation to the caller."""
    from paperless_ai.util import retrieve_with_attrition_budget
    import logging

    index = MagicMock()
    bad_retriever = MagicMock()
    bad_retriever.retrieve = MagicMock(side_effect=RuntimeError("faiss boom"))
    index.as_retriever = MagicMock(return_value=bad_retriever)

    with caplog.at_level(logging.ERROR, logger="paperless_ai.util"):
        result = retrieve_with_attrition_budget(
            global_index=index,
            query_str="anything",
            allowed_doc_ids=["1"],
            final_limit=5,
        )
    assert result.kept == []
    assert result.passes == 1
    assert any("retriever raised" in r.message for r in caplog.records)


def test_retrieve_with_attrition_budget_empty_allowed_returns_empty():
    """allowed_doc_ids=[] means every raw node fails the filter. The
    loop widens until exhaustion or max_top_k and returns empty."""
    from paperless_ai.util import retrieve_with_attrition_budget

    # Only provide enough for two passes before exhaustion plateau.
    the_50 = [_fake_scored_node(str(i)) for i in range(50)]
    nodes_by_top_k = {
        25: the_50[:25],
        100: the_50,  # plateau: still 50
    }
    index = _fake_global_index_with_retriever(nodes_by_top_k)

    result = retrieve_with_attrition_budget(
        global_index=index,
        query_str="irrelevant",
        allowed_doc_ids=[],  # nothing allowed
        final_limit=5,
    )
    assert result.kept == []
    assert result.passes == 2
