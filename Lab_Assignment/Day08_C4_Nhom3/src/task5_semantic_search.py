"""
Task 5 — Semantic Search Module.

Dense retrieval trên local JSON vector store sinh ra ở Task 4.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Tương thích với embedding model và vector store ở Task 4
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.task4_chunking_indexing import EMBEDDING_DIM, VECTOR_INDEX_PATH, get_embedding_model


@lru_cache(maxsize=1)
def _load_vector_index() -> dict[str, Any]:
    """Load local vector index from Task 4."""
    path = Path(VECTOR_INDEX_PATH)
    if not path.exists():
        return {"config": {}, "chunks": []}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        return {"config": {}, "chunks": []}
    data.setdefault("config", {})
    data.setdefault("chunks", [])
    return data


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a or not b:
        return float("-inf")

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return float("-inf")
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _index_matches_task4(index: dict[str, Any]) -> bool:
    config = index.get("config", {})
    if config.get("embedding_dim") != EMBEDDING_DIM:
        return False

    chunks = index.get("chunks", [])
    first_chunk_with_embedding = next(
        (chunk for chunk in chunks if isinstance(chunk.get("embedding"), list)),
        None,
    )
    if first_chunk_with_embedding is None:
        return False
    return len(first_chunk_with_embedding["embedding"]) == EMBEDDING_DIM


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict
        }
        Sorted by score descending.
    """
    query = query.strip()
    if not query or top_k <= 0:
        return []

    index = _load_vector_index()
    chunks = index.get("chunks", [])
    if not chunks or not _index_matches_task4(index):
        return []

    query_embedding = get_embedding_model().embed_query(query)
    scored_results = []

    for chunk in chunks:
        embedding = chunk.get("embedding")
        if not isinstance(embedding, list):
            continue

        score = _cosine_similarity(query_embedding, embedding)
        if score == float("-inf"):
            continue

        scored_results.append(
            {
                "content": chunk.get("content", ""),
                "score": score,
                "metadata": chunk.get("metadata", {}),
            }
        )

    scored_results.sort(key=lambda result: result["score"], reverse=True)
    return scored_results[:top_k]


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma túy", top_k=5)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
