import logging
import sys

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from documents.models import Document
from paperless_ai.client import AIClient
from paperless_ai.indexing import load_or_build_index
from paperless_ai.util import (
    GLOBAL_RETRIEVE_INITIAL_TOP_K,
    GLOBAL_RETRIEVE_MAX_TOP_K,
    GLOBAL_RETRIEVE_WIDEN_FACTOR,
    retrieve_with_attrition_budget,
)

logger = logging.getLogger("paperless_ai.chat")

MAX_SINGLE_DOC_CONTEXT_CHARS = 15000
SINGLE_DOC_SNIPPET_CHARS = 800
SINGLE_DOC_TOP_MATCHES = 3
MULTI_DOC_FINAL_TOP_K = 5

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about documents "
    "stored in Paperless-ngx. Answer using only the context provided "
    "in the user message below. If the context does not contain enough "
    "information to answer, say so plainly — do not fabricate facts or "
    "invent document contents."
)

USER_TEMPLATE = (
    "Context information from relevant documents:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given only the context above, answer the following question:\n"
    "\n"
    "{query_str}"
)

# Used as `refine_template` on the synthesizer. CompactAndRefine (the
# default ResponseMode.COMPACT) applies this template to follow-up
# chunks when `prompt_helper.repack` splits a long context across
# multiple LLM calls. Without a custom refine template we would fall
# back to LlamaIndex's DEFAULT_REFINE_PROMPT_SEL and lose our grounding
# system message on every chunk after the first.
REFINE_USER_TEMPLATE = (
    "Additional context information from relevant documents:\n"
    "---------------------\n"
    "{context_msg}\n"
    "---------------------\n"
    "Given only the context above (and the question and your draft "
    "answer below), refine your previous answer. If the new context "
    "does not add useful information, repeat the previous answer "
    "unchanged.\n"
    "\n"
    "Question: {query_str}\n"
    "Previous answer: {existing_answer}"
)


class _EmptyDocumentContent(Exception):
    """Raised when the single-document path has no extractable content."""


def _build_single_doc_context(doc, global_index, query_str):
    content = (doc.content or "").strip()
    if not content:
        raise _EmptyDocumentContent

    body = (
        content[:MAX_SINGLE_DOC_CONTEXT_CHARS]
        if len(content) > MAX_SINGLE_DOC_CONTEXT_CHARS
        else content
    )

    if len(content) > MAX_SINGLE_DOC_CONTEXT_CHARS:
        # Document exceeds the budget — supplement with top matching
        # snippets from the same document, retrieved through the global
        # index but filtered down to this document via the attrition
        # budget helper.
        logger.info(
            "Truncating single-document context from %s to %s characters",
            len(content),
            MAX_SINGLE_DOC_CONTEXT_CHARS,
        )
        budget_result = retrieve_with_attrition_budget(
            global_index=global_index,
            query_str=query_str,
            allowed_doc_ids=[str(doc.pk)],
            final_limit=SINGLE_DOC_TOP_MATCHES,
        )
        if budget_result.kept:
            snippets = "\n\n".join(
                f"TITLE: {n.metadata.get('title')}\n{n.text[:SINGLE_DOC_SNIPPET_CHARS]}"
                for n in budget_result.kept
            )
            body = f"{body}\n\nTOP MATCHES:\n{snippets}"

    return f"TITLE: {doc.title or doc.filename}\n{body}"


def _build_multi_doc_context(nodes_allowed):
    return "\n\n".join(
        f"TITLE: {n.metadata.get('title')}\n{n.text[:SINGLE_DOC_SNIPPET_CHARS]}"
        for n in nodes_allowed
    )


def stream_chat_with_documents(query_str: str, documents: list[Document]):
    try:
        global_index = load_or_build_index()
    except ValueError as exc:
        logger.warning("AI chat: failed to load LLM index: %s", exc)
        yield (
            "The AI index is missing or failed to load. "
            "A rebuild has been queued — please try again in a few minutes."
        )
        return

    # Empty-index guard. load_or_build_index() returns a valid index
    # object even when nothing has been indexed yet — there are just
    # no nodes in the docstore. The attrition-budget helper would
    # return an empty list in that case, which the multi-doc branch
    # would translate into the generic "no content was found"
    # fallback. That is misleading — the real reason is that nothing
    # has been indexed. Give operators a clearer message.
    if len(global_index.docstore.docs) == 0:
        yield (
            "The AI index has no content yet. "
            "Run an AI index build first."
        )
        return

    client = AIClient()

    if len(documents) == 1:
        doc = documents[0]
        try:
            context = _build_single_doc_context(doc, global_index, query_str)
        except _EmptyDocumentContent:
            yield (
                "This document has no extractable content to answer "
                "questions from."
            )
            return
    else:
        allowed_doc_ids = [str(d.pk) for d in documents]
        budget_result = retrieve_with_attrition_budget(
            global_index=global_index,
            query_str=query_str,
            allowed_doc_ids=allowed_doc_ids,
            final_limit=MULTI_DOC_FINAL_TOP_K,
        )
        if not budget_result.kept:
            logger.debug(
                "chat multi-doc empty result: final_top_k=%s "
                "raw_returned=%s allowed_kept=%s allowed_doc_ids_count=%s "
                "passes=%s",
                budget_result.final_top_k,
                budget_result.raw_returned,
                len(budget_result.kept),
                len(allowed_doc_ids),
                budget_result.passes,
            )
            yield "No content was found to answer your question."
            return
        context = _build_multi_doc_context(budget_result.kept)

    # Synthetic-node workaround. ResponseSynthesizer short-circuits on
    # empty nodes AND overrides any pre-populated `context_str` in the
    # template when real nodes are passed. To feed our hand-built
    # context string through, we wrap it in a single TextNode with
    # empty metadata (non-empty metadata gets prepended via
    # `text_template.format(content=..., metadata_str=...)`).
    context_node = NodeWithScore(
        node=TextNode(text=context, metadata={}),
        score=1.0,
    )

    qa_template = ChatPromptTemplate([
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=USER_TEMPLATE),
    ])
    refine_template = ChatPromptTemplate([
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=REFINE_USER_TEMPLATE),
    ])

    synthesizer = get_response_synthesizer(
        llm=client.llm,
        text_qa_template=qa_template,
        refine_template=refine_template,
        response_mode=ResponseMode.COMPACT,
        streaming=True,
    )

    try:
        response = synthesizer.synthesize(
            query=QueryBundle(query_str=query_str),
            nodes=[context_node],
        )
    except Exception:
        logger.exception("AI chat: synthesizer.synthesize raised")
        yield (
            "AI backend is unavailable right now — please try again later."
        )
        return

    try:
        for token in response.response_gen:
            yield token
            sys.stdout.flush()
    except Exception:
        logger.exception("AI chat: response stream raised mid-yield")
        yield "\n\n[response interrupted by an error]"
        return
