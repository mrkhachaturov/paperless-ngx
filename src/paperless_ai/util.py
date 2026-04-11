"""
Shared helpers for the paperless_ai chat path.

Keep this file pure — no Django, no HTTP, no heavy LlamaIndex imports
at module level. Downstream code depends on these helpers being cheap
to import and easy to unit-test without fixtures.
"""

import logging
from dataclasses import dataclass, field
from typing import Iterable, List

logger = logging.getLogger("paperless_ai.util")

# Tunables for retrieve_with_attrition_budget. They are module-level
# so both the helper and its callers (chat.py) can reference them by
# name for tests and for consistency.
GLOBAL_RETRIEVE_INITIAL_TOP_K = 25
GLOBAL_RETRIEVE_MAX_TOP_K = 500
GLOBAL_RETRIEVE_WIDEN_FACTOR = 4


@dataclass
class AttritionBudgetResult:
    """
    Return shape for ``retrieve_with_attrition_budget``. Exposes the
    filtered nodes alongside the last-pass statistics so callers can
    log them without re-deriving state from the retriever.
    """
    kept: List = field(default_factory=list)
    final_top_k: int = 0   # the similarity_top_k of the final pass
    raw_returned: int = 0  # number of raw nodes returned on the final pass
    passes: int = 0        # number of retrieve() calls made


def retrieve_with_attrition_budget(
    global_index,
    query_str: str,
    allowed_doc_ids,
    final_limit: int,
    initial_top_k: int = GLOBAL_RETRIEVE_INITIAL_TOP_K,
    max_top_k: int = GLOBAL_RETRIEVE_MAX_TOP_K,
    widen_factor: int = GLOBAL_RETRIEVE_WIDEN_FACTOR,
) -> AttritionBudgetResult:
    """
    Retrieve nodes from the global index and permission-filter down to
    ``allowed_doc_ids``. If the first pass produces fewer than
    ``final_limit`` allowed nodes, widen ``similarity_top_k`` by
    ``widen_factor`` and retry, up to ``max_top_k`` or until the index
    stops producing new raw nodes. Returns an ``AttritionBudgetResult``
    with at most ``final_limit`` allowed nodes in ``kept`` plus the
    last-pass statistics so callers can log a full diagnostic line.

    Does NOT cap at ``len(docstore.docs)``: stale FAISS vectors from
    deleted documents silently drop during docstore fetch, so the
    docstore count is a lower bound on live retrievable nodes, not an
    upper bound on what FAISS tries to return. Exhaustion is detected
    empirically — if two consecutive passes return the same raw node
    count, widening further will not help and we stop.

    Rationale: the vendored FAISS integration does not honour
    ``doc_ids`` at query time, so we filter in Python after retrieval.
    When the caller's allowed set is narrow relative to the full
    index, the initial ``top_k`` may contain zero or too few allowed
    matches. Widening gives the permission filter room to find
    allowed matches that happened to rank below the initial
    threshold.
    """
    result = AttritionBudgetResult()
    top_k = initial_top_k
    prev_raw_count = -1

    while True:
        retriever = global_index.as_retriever(similarity_top_k=top_k)
        result.passes += 1
        result.final_top_k = top_k
        try:
            nodes_raw = retriever.retrieve(query_str)
        except Exception:
            logger.exception(
                "retrieve_with_attrition_budget: retriever raised at top_k=%s",
                top_k,
            )
            # result.kept stays as whatever the prior pass produced
            return result

        result.raw_returned = len(nodes_raw)
        result.kept = filter_nodes_by_doc_ids(
            nodes_raw, doc_ids=allowed_doc_ids, limit=final_limit,
        )
        if len(result.kept) >= final_limit:
            return result

        # Exhaustion detector: widening did not return more raw nodes,
        # or the index returned fewer nodes than we asked for (FAISS
        # has nothing more to give regardless of top_k).
        if len(nodes_raw) == prev_raw_count or len(nodes_raw) < top_k:
            return result
        prev_raw_count = len(nodes_raw)

        if top_k >= max_top_k:
            return result
        top_k = min(top_k * widen_factor, max_top_k)


def filter_nodes_by_doc_ids(nodes: Iterable, doc_ids, limit: int) -> List:
    """
    Keep only nodes whose metadata['document_id'] is in the allowed set,
    preserving input order, truncated to ``limit`` entries.

    Accepts either bare ``BaseNode`` objects or ``NodeWithScore``
    wrappers, detected by the presence of a ``.node`` attribute. In
    both cases the returned objects are the original wrappers — this
    helper does not unwrap. That way callers that want to surface the
    similarity score later (e.g. the deferred citations iteration)
    still have access to it.
    """
    allowed = set(doc_ids)
    kept: List = []
    for n in nodes:
        node = n.node if hasattr(n, "node") else n
        if node.metadata.get("document_id") in allowed:
            kept.append(n)
            if len(kept) >= limit:
                break
    return kept
