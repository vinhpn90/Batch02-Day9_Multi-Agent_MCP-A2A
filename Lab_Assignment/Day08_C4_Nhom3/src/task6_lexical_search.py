"""
Task 6 — Lexical Search Module (BM25).

BM25 keyword search trên cùng local JSON vector store sinh ra ở Task 4.
Implementation này dùng Python chuẩn để không phụ thuộc runtime vào package
ngoài, nhưng vẫn theo công thức BM25Okapi với k1=1.5 và b=0.75.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.task4_chunking_indexing import VECTOR_INDEX_PATH

TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
BM25_K1 = 1.5
BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản cho tiếng Việt: lowercase và giữ chữ có dấu."""
    return TOKEN_PATTERN.findall(text.lower())


@lru_cache(maxsize=1)
def _load_corpus() -> list[dict]:
    path = Path(VECTOR_INDEX_PATH)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    chunks = data.get("chunks", []) if isinstance(data, dict) else []
    corpus = []
    for chunk in chunks:
        content = chunk.get("content", "")
        if not content:
            continue
        corpus.append(
            {
                "content": content,
                "metadata": chunk.get("metadata", {}),
            }
        )
    return corpus


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [_tokenize(doc.get("content", "")) for doc in corpus]
    doc_freq: Counter[str] = Counter()
    term_freqs: list[Counter[str]] = []
    doc_lengths: list[int] = []

    for tokens in tokenized_corpus:
        term_freq = Counter(tokens)
        term_freqs.append(term_freq)
        doc_lengths.append(len(tokens))
        doc_freq.update(term_freq.keys())

    doc_count = len(corpus)
    avg_doc_length = sum(doc_lengths) / doc_count if doc_count else 0.0
    idf = {
        term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
        for term, freq in doc_freq.items()
    }

    return {
        "corpus": corpus,
        "term_freqs": term_freqs,
        "doc_lengths": doc_lengths,
        "avg_doc_length": avg_doc_length,
        "idf": idf,
    }


@lru_cache(maxsize=1)
def _get_bm25_index():
    return build_bm25_index(_load_corpus())


def _bm25_score(query_tokens: list[str], index: dict[str, Any], doc_index: int) -> float:
    term_freq = index["term_freqs"][doc_index]
    doc_length = index["doc_lengths"][doc_index]
    avg_doc_length = index["avg_doc_length"]
    if not doc_length or not avg_doc_length:
        return 0.0

    score = 0.0
    for term in query_tokens:
        frequency = term_freq.get(term, 0)
        if frequency == 0:
            continue

        idf = index["idf"].get(term, 0.0)
        denominator = frequency + BM25_K1 * (
            1 - BM25_B + BM25_B * doc_length / avg_doc_length
        )
        score += idf * (frequency * (BM25_K1 + 1)) / denominator
    return score


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

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
    query_tokens = _tokenize(query)
    if not query_tokens or top_k <= 0:
        return []

    index = _get_bm25_index()
    corpus = index["corpus"]
    if not corpus:
        return []

    scored = []
    for doc_index, doc in enumerate(corpus):
        score = _bm25_score(query_tokens, index, doc_index)
        if score <= 0:
            continue
        scored.append(
            {
                "content": doc["content"],
                "score": float(score),
                "metadata": doc.get("metadata", {}),
            }
        )

    scored.sort(key=lambda result: result["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma túy", top_k=5)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
