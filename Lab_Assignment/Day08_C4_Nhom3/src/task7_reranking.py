"""
Task 7 — Reranking Module.

Default method: cross-encoder reranking with
`jinaai/jina-reranker-v2-base-multilingual`.

The local Jina model is used when it is available in the Hugging Face cache.
If it cannot be loaded, reranking falls back to a deterministic lexical overlap
score so retrieval code and tests still run without network access.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
RERANKER_MAX_LENGTH = 1024
TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@lru_cache(maxsize=1)
def _load_jina_reranker():
    """
    Load the selected Jina reranker from local cache.

    `local_files_only=True` keeps this module offline-friendly. Download the
    model beforehand if real cross-encoder scores are required.
    """
    try:
        import torch
        import transformers.models.xlm_roberta.modeling_xlm_roberta as xlm_roberta_modeling
        from transformers import AutoModelForSequenceClassification

        if not hasattr(xlm_roberta_modeling, "create_position_ids_from_input_ids"):
            def create_position_ids_from_input_ids(
                input_ids,
                padding_idx,
                past_key_values_length=0,
            ):
                mask = input_ids.ne(padding_idx).int()
                incremental_indices = (
                    torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
                ) * mask
                return incremental_indices.long() + padding_idx

            xlm_roberta_modeling.create_position_ids_from_input_ids = (
                create_position_ids_from_input_ids
            )

        model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL,
            trust_remote_code=True,
            local_files_only=True,
        )
        model.eval()
        return model
    except Exception:
        return None


def _fallback_score(query: str, content: str, original_score: float = 0.0) -> float:
    query_tokens = set(_tokenize(query))
    content_tokens = _tokenize(content)
    if not query_tokens or not content_tokens:
        return float(original_score)

    content_token_set = set(content_tokens)
    overlap = len(query_tokens & content_token_set) / len(query_tokens)
    term_frequency = sum(1 for token in content_tokens if token in query_tokens)
    length_norm = math.sqrt(len(content_tokens))
    return overlap + (term_frequency / length_norm) + 0.01 * float(original_score)


def _score_with_jina_model(query: str, candidates: list[dict]) -> list[float] | None:
    model = _load_jina_reranker()
    if model is None:
        return None

    pairs = [[query, candidate.get("content", "")] for candidate in candidates]
    try:
        scores = model.compute_score(pairs, max_length=RERANKER_MAX_LENGTH)
    except TypeError:
        scores = model.compute_score(pairs)
    except Exception:
        return None

    if isinstance(scores, (float, int)):
        return [float(scores)]
    return [float(score) for score in scores]


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank score descending.
    """
    if not query.strip() or top_k <= 0 or not candidates:
        return []

    scores = _score_with_jina_model(query, candidates)
    method = "jina_cross_encoder" if scores and len(scores) == len(candidates) else "lexical_fallback"
    if method == "lexical_fallback":
        scores = [
            _fallback_score(query, candidate.get("content", ""), candidate.get("score", 0.0))
            for candidate in candidates
        ]

    reranked = []
    for candidate, score in zip(candidates, scores):
        item = candidate.copy()
        metadata = item.get("metadata", {})
        item["metadata"] = metadata.copy() if isinstance(metadata, dict) else {}
        item["metadata"]["rerank_model"] = RERANKER_MODEL
        item["metadata"]["rerank_method"] = method
        item["original_score"] = float(candidate.get("score", 0.0))
        item["score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = lambda * sim(query, doc) - (1-lambda) * max(sim(doc, selected_docs))
    """
    if top_k <= 0 or not candidates:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            embedding = candidates[idx].get("embedding", [])
            relevance = _cosine_similarity(query_embedding, embedding)
            max_sim_to_selected = max(
                (
                    _cosine_similarity(embedding, candidates[selected_idx].get("embedding", []))
                    for selected_idx in selected
                ),
                default=0.0,
            )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = candidates[idx].copy()
        item["score"] = float(item.get("score", 0.0))
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = sum(1 / (k + rank_r(d)))
    """
    if top_k <= 0:
        return []

    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("content", "")
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)
    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    if method == "rrf":
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content']}")
