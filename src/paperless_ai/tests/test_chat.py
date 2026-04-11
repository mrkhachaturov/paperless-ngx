"""
Unit tests for paperless_ai.chat.

These tests mock every LlamaIndex object — `load_or_build_index`,
`get_response_synthesizer`, and `AIClient`. They do not construct a
real index, so they are fast and work regardless of which embedding
backend is configured. The tests assert on what the generator yields
and on what arguments were passed to the synthesizer — never on any
real LLM output.
"""

from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.base.llms.types import MessageRole
from llama_index.core.schema import NodeWithScore, TextNode


def _mock_response_stream(chunks):
    """Build a MagicMock that mimics StreamingResponse."""
    response = MagicMock()
    response.response_gen = iter(chunks)
    response.source_nodes = []
    return response


def _mock_scored_node(document_id: str, text: str, title: str = None):
    node = TextNode(
        text=text,
        metadata={"document_id": document_id, "title": title or f"Doc {document_id}"},
    )
    return NodeWithScore(node=node, score=0.9)


@pytest.fixture
def mock_doc():
    doc = MagicMock()
    doc.pk = 42
    doc.title = "Test Document"
    doc.filename = "test.pdf"
    doc.content = "This is the body of the test document."
    return doc


@pytest.fixture
def mock_long_doc():
    doc = MagicMock()
    doc.pk = 99
    doc.title = "Long Document"
    doc.filename = "long.pdf"
    doc.content = "x" * 20000
    return doc


def _patch_chat_dependencies(*, index=None, synth_response=None, raise_load=False):
    """
    Context manager helper: patches load_or_build_index, AIClient, and
    get_response_synthesizer. Returns the tuple of mocks so tests can
    inspect call arguments.

    The default mock `index` is always seeded with a non-empty
    `docstore.docs` dict so the empty-index guard in
    `stream_chat_with_documents` does not fire. Tests that want to
    exercise the empty-index fallback must pass an explicit index
    with `.docstore.docs = {}`.
    """
    from contextlib import ExitStack

    stack = ExitStack()

    mock_load = stack.enter_context(patch("paperless_ai.chat.load_or_build_index"))
    if raise_load:
        mock_load.side_effect = ValueError("index missing")
    else:
        if index is None:
            index = MagicMock()
            # Seed a non-empty docstore so the empty-index guard passes.
            index.docstore.docs = {"default-node-1": MagicMock()}
        else:
            # Caller supplied an index. If they did not explicitly set
            # docstore.docs, seed it too — only the dedicated
            # empty-index test expects a zero-length docstore and it
            # passes its own index with docs={}.
            docs_attr = getattr(index.docstore, "docs", None)
            if not isinstance(docs_attr, dict):
                index.docstore.docs = {"default-node-1": MagicMock()}
        mock_load.return_value = index

    mock_client_cls = stack.enter_context(patch("paperless_ai.chat.AIClient"))
    mock_client = MagicMock()
    mock_client.llm = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_synth_factory = stack.enter_context(
        patch("paperless_ai.chat.get_response_synthesizer")
    )
    mock_synth = MagicMock()
    mock_synth_factory.return_value = mock_synth
    mock_synth.synthesize.return_value = synth_response or _mock_response_stream([])

    return stack, mock_load, mock_client_cls, mock_synth_factory, mock_synth


# ─── Structural tests for the new serving path ──────────────────────

def test_stream_chat_does_not_construct_local_vector_store_index(mock_doc):
    """No temporary VectorStoreIndex should be built on a chat request."""
    from paperless_ai.chat import stream_chat_with_documents

    stack, *_, mock_synth = _patch_chat_dependencies(
        synth_response=_mock_response_stream(["hello"]),
    )
    with stack, patch(
        "llama_index.core.VectorStoreIndex",
    ) as mock_vsi:
        list(stream_chat_with_documents("What is this?", [mock_doc]))
        mock_vsi.assert_not_called()


def test_stream_chat_passes_raw_query_to_synthesizer(mock_doc):
    """The QueryBundle given to synthesize.query must match query_str."""
    from paperless_ai.chat import stream_chat_with_documents

    stack, *_, mock_synth = _patch_chat_dependencies(
        synth_response=_mock_response_stream(["answer"]),
    )
    with stack:
        list(stream_chat_with_documents("specific question", [mock_doc]))

    call = mock_synth.synthesize.call_args
    query_bundle = call.kwargs.get("query") or call.args[0]
    assert query_bundle.query_str == "specific question"


def test_stream_chat_passes_single_synthetic_context_node_to_synthesizer(mock_doc):
    """Exactly one NodeWithScore whose inner text is our hand-built context,
    empty metadata (so nothing bleeds into the prompt), and score=1.0."""
    from paperless_ai.chat import stream_chat_with_documents

    stack, *_, mock_synth = _patch_chat_dependencies(
        synth_response=_mock_response_stream(["answer"]),
    )
    with stack:
        list(stream_chat_with_documents("What is this?", [mock_doc]))

    call = mock_synth.synthesize.call_args
    nodes = call.kwargs.get("nodes")
    assert len(nodes) == 1
    wrapper = nodes[0]
    assert isinstance(wrapper, NodeWithScore)
    assert wrapper.score == 1.0
    inner = wrapper.node
    assert isinstance(inner, TextNode)
    assert "TITLE: Test Document" in inner.text
    assert "This is the body of the test document." in inner.text
    # IMPORTANT: metadata must be empty so TextNode.get_content does not
    # prepend "source: foo\n\n" before our context.
    assert inner.metadata == {}


def test_stream_chat_synthesizer_never_called_with_empty_nodes(mock_doc):
    """Empty nodes short-circuits to 'Empty Response' — we never want
    that to happen from our generator under any branch."""
    from paperless_ai.chat import stream_chat_with_documents

    stack, *_, mock_synth = _patch_chat_dependencies(
        synth_response=_mock_response_stream(["answer"]),
    )
    with stack:
        list(stream_chat_with_documents("question", [mock_doc]))

    if mock_synth.synthesize.called:
        nodes = mock_synth.synthesize.call_args.kwargs.get("nodes")
        assert len(nodes) >= 1, "synthesize was called with empty nodes"


def test_stream_chat_uses_system_role_message(mock_doc):
    """The synthesizer must receive a ChatPromptTemplate whose messages
    include a SYSTEM-role entry with the grounding instruction."""
    from paperless_ai.chat import stream_chat_with_documents

    stack, *_, mock_synth_factory, _ = _patch_chat_dependencies(
        synth_response=_mock_response_stream(["answer"]),
    )
    with stack:
        list(stream_chat_with_documents("question", [mock_doc]))

    factory_call = mock_synth_factory.call_args
    qa_template = factory_call.kwargs["text_qa_template"]
    # ChatPromptTemplate has a `message_templates` attribute
    system_messages = [
        m for m in qa_template.message_templates if m.role == MessageRole.SYSTEM
    ]
    assert len(system_messages) == 1
    assert "only" in system_messages[0].content.lower()


def test_stream_chat_passes_refine_template_too(mock_doc):
    """Chain A must pass a refine_template with the same grounding, not
    rely on the default DEFAULT_REFINE_PROMPT_SEL."""
    from paperless_ai.chat import stream_chat_with_documents

    stack, *_, mock_synth_factory, _ = _patch_chat_dependencies(
        synth_response=_mock_response_stream(["answer"]),
    )
    with stack:
        list(stream_chat_with_documents("question", [mock_doc]))

    factory_call = mock_synth_factory.call_args
    refine_template = factory_call.kwargs["refine_template"]
    system_messages = [
        m for m in refine_template.message_templates
        if m.role == MessageRole.SYSTEM
    ]
    assert len(system_messages) == 1


# ─── Single-document path tests ──────────────────────────────────────

def test_stream_chat_single_doc_under_budget_skips_retrieve(mock_doc):
    """Short content — no retrieval, synthesizer still runs."""
    from paperless_ai.chat import stream_chat_with_documents

    stack, mock_load, *_, mock_synth = _patch_chat_dependencies(
        synth_response=_mock_response_stream(["answer"]),
    )
    with stack:
        list(stream_chat_with_documents("question", [mock_doc]))

    # The global_index's as_retriever was never called because the
    # single-doc path short-circuits the retrieve helper.
    mock_load.return_value.as_retriever.assert_not_called()


def test_stream_chat_single_doc_over_budget_uses_attrition_budget_filtered_to_one_doc(
    mock_long_doc,
):
    """Long content — retrieve runs, allowed_doc_ids scoped to that one doc."""
    from paperless_ai.chat import stream_chat_with_documents

    index = MagicMock()
    # Seed docstore so empty-index guard passes. The helper would do
    # this for us, but we are constructing a custom index here.
    index.docstore.docs = {"node-1": MagicMock()}
    # Mix: one chunk from the target doc, several from other docs.
    raw_nodes = [
        _mock_scored_node("99", "chunk from our target doc", "Long Document"),
        _mock_scored_node("77", "other doc noise"),
        _mock_scored_node("88", "more noise"),
    ]
    index.as_retriever.return_value.retrieve.return_value = raw_nodes

    stack, *_, mock_synth = _patch_chat_dependencies(
        index=index,
        synth_response=_mock_response_stream(["answer"]),
    )
    with stack:
        list(stream_chat_with_documents("question", [mock_long_doc]))

    # The synthetic-node context must contain ONLY the target doc's
    # chunk under "TOP MATCHES", not the other two docs.
    call = mock_synth.synthesize.call_args
    context_text = call.kwargs["nodes"][0].node.text
    assert "chunk from our target doc" in context_text
    assert "other doc noise" not in context_text
    assert "more noise" not in context_text


def test_stream_chat_empty_single_doc_yields_fallback(mock_doc):
    """Single doc with empty content — specific fallback, no synthesis."""
    from paperless_ai.chat import stream_chat_with_documents

    mock_doc.content = "   "  # whitespace only, strips to empty
    stack, *_, mock_synth = _patch_chat_dependencies()
    with stack:
        output = list(stream_chat_with_documents("question", [mock_doc]))

    assert any("no extractable content" in chunk for chunk in output)
    mock_synth.synthesize.assert_not_called()


# ─── Multi-document path tests ───────────────────────────────────────

def test_stream_chat_permission_filter_drops_disallowed_nodes():
    """Multi-doc branch: raw retrieve returns 10 nodes, 3 of which match
    allowed document_ids. The hand-built context must contain exactly
    those 3 titles."""
    from paperless_ai.chat import stream_chat_with_documents

    docs = [MagicMock(pk=1), MagicMock(pk=2), MagicMock(pk=3)]
    for d in docs:
        d.title = f"Doc {d.pk}"

    index = MagicMock()
    index.docstore.docs = {"node-1": MagicMock()}
    raw_nodes = [
        _mock_scored_node("99", "disallowed A"),
        _mock_scored_node("1", "allowed 1 chunk", "Doc 1"),
        _mock_scored_node("98", "disallowed B"),
        _mock_scored_node("2", "allowed 2 chunk", "Doc 2"),
        _mock_scored_node("97", "disallowed C"),
        _mock_scored_node("3", "allowed 3 chunk", "Doc 3"),
        _mock_scored_node("96", "disallowed D"),
        _mock_scored_node("95", "disallowed E"),
        _mock_scored_node("94", "disallowed F"),
        _mock_scored_node("93", "disallowed G"),
    ]
    index.as_retriever.return_value.retrieve.return_value = raw_nodes

    stack, *_, mock_synth = _patch_chat_dependencies(
        index=index,
        synth_response=_mock_response_stream(["answer"]),
    )
    with stack:
        list(stream_chat_with_documents("question", docs))

    call = mock_synth.synthesize.call_args
    context_text = call.kwargs["nodes"][0].node.text
    assert "allowed 1 chunk" in context_text
    assert "allowed 2 chunk" in context_text
    assert "allowed 3 chunk" in context_text
    for disallowed in ["disallowed A", "disallowed B", "disallowed C"]:
        assert disallowed not in context_text


def test_stream_chat_attrition_to_zero_yields_fallback():
    """Multi-doc branch with no allowed matches anywhere — single fallback."""
    from paperless_ai.chat import stream_chat_with_documents

    docs = [MagicMock(pk=1), MagicMock(pk=2)]
    for d in docs:
        d.title = f"Doc {d.pk}"

    index = MagicMock()
    index.docstore.docs = {"node-1": MagicMock()}
    # Exhaustion plateau: all passes return the same raw nodes, none allowed.
    raw_nodes = [_mock_scored_node("999", "unrelated") for _ in range(10)]
    index.as_retriever.return_value.retrieve.return_value = raw_nodes

    stack, *_, mock_synth = _patch_chat_dependencies(index=index)
    with stack:
        output = list(stream_chat_with_documents("question", docs))

    assert any("No content was found" in chunk for chunk in output)
    mock_synth.synthesize.assert_not_called()


# ─── Error-handling tests ────────────────────────────────────────────

def test_stream_chat_index_load_failure_yields_fallback(mock_doc):
    """load_or_build_index raises ValueError — generator catches and
    yields the specific operator-facing message."""
    from paperless_ai.chat import stream_chat_with_documents

    stack, *_, mock_synth = _patch_chat_dependencies(raise_load=True)
    with stack:
        output = list(stream_chat_with_documents("question", [mock_doc]))

    assert any("index is missing or failed to load" in chunk for chunk in output)
    mock_synth.synthesize.assert_not_called()


def test_stream_chat_synthesizer_exception_yields_fallback(mock_doc):
    """synthesizer.synthesize raising yields the AI-backend fallback."""
    from paperless_ai.chat import stream_chat_with_documents

    stack, *_, mock_synth = _patch_chat_dependencies()
    mock_synth.synthesize.side_effect = RuntimeError("boom")
    with stack:
        output = list(stream_chat_with_documents("question", [mock_doc]))

    assert any("AI backend is unavailable" in chunk for chunk in output)


def test_stream_chat_response_stream_mid_yield_exception_yields_interrupted_marker(
    mock_doc,
):
    """response.response_gen raises partway through — the generator
    must yield the interrupted-marker text and return cleanly."""
    from paperless_ai.chat import stream_chat_with_documents

    def exploding_gen():
        yield "partial answer "
        raise RuntimeError("mid-stream boom")

    response = MagicMock()
    response.response_gen = exploding_gen()
    response.source_nodes = []

    stack, *_, mock_synth = _patch_chat_dependencies(synth_response=response)
    with stack:
        output = list(stream_chat_with_documents("question", [mock_doc]))

    # First chunk is the partial answer, last chunk is the interrupted marker.
    assert "partial answer " in output
    assert any("[response interrupted by an error]" in c for c in output)


def test_stream_chat_empty_index_yields_fallback(mock_doc):
    """global_index.docstore.docs is empty — yield the 'no content yet'
    fallback before any AIClient or synthesizer call."""
    from paperless_ai.chat import stream_chat_with_documents

    empty_index = MagicMock()
    empty_index.docstore.docs = {}

    stack, mock_load, mock_client_cls, mock_synth_factory, mock_synth = (
        _patch_chat_dependencies(index=empty_index)
    )
    with stack:
        output = list(stream_chat_with_documents("question", [mock_doc]))

    assert any("index has no content yet" in chunk for chunk in output)
    mock_synth.synthesize.assert_not_called()


def test_stream_chat_attrition_widening_through_chat_layer():
    """Multi-doc: first retrieve pass (top_k=25) returns zero allowed,
    second pass (top_k=100) returns MULTI_DOC_FINAL_TOP_K=5 allowed.
    Chat must end up with a normal synthesized answer, not the fallback.

    Pass 2 must return at least 5 allowed nodes because the helper
    keeps widening until final_limit is reached. If pass 2 returned
    only 3, the helper would widen to top_k=400 for a third pass."""
    from paperless_ai.chat import stream_chat_with_documents

    docs = [MagicMock(pk=i) for i in range(1, 6)]
    for d in docs:
        d.title = f"Doc {d.pk}"

    # Build an index whose retriever returns different nodes depending
    # on the similarity_top_k it was constructed with.
    index = MagicMock()
    # docstore non-empty so the empty-index guard passes
    index.docstore.docs = {f"node-{i}": MagicMock() for i in range(200)}

    def as_retriever(similarity_top_k):
        retriever = MagicMock()
        if similarity_top_k == 25:
            # All disallowed on the first pass.
            retriever.retrieve.return_value = [
                _mock_scored_node("99", f"noise {i}") for i in range(25)
            ]
        else:
            # Second pass (top_k=100) returns 95 disallowed + 5 allowed.
            # The helper exits at len(kept) >= 5, so no third pass.
            retriever.retrieve.return_value = [
                _mock_scored_node("99", f"noise {i}") for i in range(95)
            ] + [
                _mock_scored_node(str(pk), f"allowed {pk}", f"Doc {pk}")
                for pk in (1, 2, 3, 4, 5)
            ]
        return retriever

    index.as_retriever.side_effect = as_retriever

    stack, *_, mock_synth = _patch_chat_dependencies(
        index=index,
        synth_response=_mock_response_stream(["answer"]),
    )
    with stack:
        output = list(stream_chat_with_documents("question", docs))

    # Widening succeeded — synthesizer was called and we got a real answer.
    assert "answer" in output
    mock_synth.synthesize.assert_called_once()
    # Two retriever constructions: first at 25, then at 100.
    top_k_calls = [
        c.kwargs["similarity_top_k"] if c.kwargs else c.args[0]
        for c in index.as_retriever.call_args_list
    ]
    assert top_k_calls == [25, 100]
