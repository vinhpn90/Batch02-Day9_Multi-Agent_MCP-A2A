"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def _with_source(results: list[dict], source: str) -> list[dict]:
    normalized = []
    for result in results:
        item = result.copy()
        metadata = item.get("metadata", {})
        item["metadata"] = metadata.copy() if isinstance(metadata, dict) else {}
        item["source"] = source
        normalized.append(item)
    return normalized


def _fallback_pageindex(query: str, top_k: int) -> list[dict]:
    return pageindex_search(query, top_k=top_k)


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    query = query.strip()
    if not query or top_k <= 0:
        return []

    retrieval_k = max(top_k * 3, top_k)

    try:
        dense_results = semantic_search(query, top_k=retrieval_k)
    except Exception:
        dense_results = []

    try:
        sparse_results = lexical_search(query, top_k=retrieval_k)
    except Exception:
        sparse_results = []

    merged = rerank_rrf([dense_results, sparse_results], top_k=retrieval_k)
    merged = _with_source(merged, "hybrid")

    if use_reranking and merged:
        try:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
            final_results = _with_source(final_results, "hybrid")
        except Exception:
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        return _fallback_pageindex(query, top_k=top_k)

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
