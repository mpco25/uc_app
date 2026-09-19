"""Hybrid keyword and semantic retrieval using rank fusion."""

from search.keyword import keyword_search
from search.semantic import semantic_search
from config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    REQUEST_TIMEOUT,
)

RRF_CONSTANT = 60


def reciprocal_rank(rank, constant=RRF_CONSTANT):
    """Return the Reciprocal Rank Fusion contribution for one rank."""
    return 1.0 / (constant + rank)


def hybrid_search(
    index,
    metadata,
    question,
    top_n,
    semantic_candidate_multiplier=2,
    minimum_semantic_candidates=50,
    ollama_base_url=OLLAMA_BASE_URL,
    embedding_model=EMBEDDING_MODEL,
    request_timeout=REQUEST_TIMEOUT,
):
    """
    Combine keyword and semantic retrieval with Reciprocal Rank Fusion.

    Keyword and semantic results are generated independently. Records that
    rank well in both lists receive the strongest combined score.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    semantic_candidate_count = min(
        index.ntotal,
        max(
            top_n * semantic_candidate_multiplier,
            minimum_semantic_candidates,
        ),
    )

    semantic_results = semantic_search(
        index=index,
        metadata=metadata,
        question=question,
        top_n=semantic_candidate_count,
        ollama_base_url=ollama_base_url,
        embedding_model=embedding_model,
        request_timeout=request_timeout,
    )

    keyword_results = keyword_search(
        metadata=metadata,
        question=question,
    )

    merged = {}

    for rank, result in enumerate(semantic_results, start=1):
        position = result["position"]
        item = dict(result)
        item["semantic_rank"] = rank
        item["hybrid_score"] = reciprocal_rank(rank)
        merged[position] = item

    for rank, result in enumerate(keyword_results, start=1):
        position = result["position"]
        keyword_rrf = reciprocal_rank(rank)

        if position in merged:
            item = merged[position]
            item["keyword_rank"] = rank
            item["hybrid_score"] += keyword_rrf
            item["lexical_score"] = result["lexical_score"]
            item["matched_terms"] = result.get("matched_terms", [])
            item["match_locations"] = result.get("match_locations", [])
            item["sources"] = ["semantic", "keyword"]
            item["source"] = "hybrid"
        else:
            item = dict(result)
            item["keyword_rank"] = rank
            item["hybrid_score"] = keyword_rrf
            item["source"] = "hybrid"
            merged[position] = item

    combined_results = sorted(
        merged.values(),
        key=lambda result: (
            result["hybrid_score"],
            result.get("lexical_score", 0),
            result.get("similarity") or 0.0,
        ),
        reverse=True,
    )

    return combined_results[:top_n]
