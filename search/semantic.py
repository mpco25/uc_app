"""Ollama embedding generation and FAISS semantic retrieval."""

import faiss
import httpx
import numpy as np


from config import (
    OLLAMA_BASE_URL,
    REQUEST_TIMEOUT,
    EMBEDDING_MODEL,
)


def create_embedding(
    text,
    ollama_base_url=OLLAMA_BASE_URL,
    embedding_model=EMBEDDING_MODEL,
    request_timeout=REQUEST_TIMEOUT,
):
    """Create and L2-normalize one embedding through Ollama."""
    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError("Cannot create an embedding for empty text.")

    response = httpx.post(
        f"{ollama_base_url}/api/embed",
        json={
            "model": embedding_model,
            "input": cleaned_text,
            "truncate": True,
            "keep_alive": "10m",
        },
        timeout=request_timeout,
    )
    response.raise_for_status()

    embeddings = response.json().get("embeddings")

    if not embeddings:
        raise RuntimeError("Ollama returned no embedding.")

    vector = np.asarray([embeddings[0]], dtype=np.float32)

    if vector.ndim != 2:
        raise RuntimeError(f"Unexpected embedding shape: {vector.shape}")

    faiss.normalize_L2(vector)
    return vector


def semantic_search(
    index,
    metadata,
    question,
    top_n,
    ollama_base_url=OLLAMA_BASE_URL,
    embedding_model=EMBEDDING_MODEL,
    request_timeout=REQUEST_TIMEOUT,
):
    """Retrieve the top-N semantically similar table records."""
    if index.ntotal != len(metadata):
        raise ValueError(
            f"FAISS contains {index.ntotal} vectors, "
            f"but metadata contains {len(metadata)} records."
        )

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    question_embedding = create_embedding(
        question,
        ollama_base_url=ollama_base_url,
        embedding_model=embedding_model,
        request_timeout=request_timeout,
    )

    similarities, positions = index.search(
        question_embedding,
        min(top_n, index.ntotal),
    )

    results = []

    for rank, (similarity, position) in enumerate(
        zip(similarities[0], positions[0], strict=True),
        start=1,
    ):
        position = int(position)

        if position < 0:
            continue

        if position >= len(metadata):
            raise IndexError(
                f"FAISS returned position {position}, but metadata "
                f"contains only {len(metadata)} records."
            )

        results.append(
            {
                "position": position,
                "source": "semantic",
                "sources": ["semantic"],
                "similarity": float(similarity),
                "semantic_rank": rank,
                "keyword_rank": None,
                "lexical_score": 0,
                "hybrid_score": None,
                "matched_terms": [],
                "match_locations": [],
                "metadata": metadata[position],
            }
        )

    return results
