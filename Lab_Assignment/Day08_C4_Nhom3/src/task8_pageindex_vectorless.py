"""
Task 8 — PageIndex Vectorless RAG.

PageIndex là fallback retrieval không phụ thuộc vào vector store. Module này
cố gắng dùng PageIndex SDK nếu package/API key đã sẵn sàng; khi môi trường chưa
có SDK hoặc chưa upload tài liệu, nó fallback sang keyword retrieval local trên
chunks của Task 4 để pipeline vẫn chạy ổn.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
PAGEINDEX_MANIFEST_PATH = INDEX_DIR / "pageindex_upload_manifest.json"


def _document_type(path: Path) -> str:
    relative = path.relative_to(STANDARDIZED_DIR)
    return relative.parts[0] if len(relative.parts) > 1 else "unknown"


def _format_pageindex_result(result: Any, fallback_metadata: dict | None = None) -> dict:
    content = (
        getattr(result, "text", None)
        or getattr(result, "content", None)
        or (result.get("text") if isinstance(result, dict) else None)
        or (result.get("content") if isinstance(result, dict) else None)
        or ""
    )
    score = (
        getattr(result, "score", None)
        or getattr(result, "relevance_score", None)
        or (result.get("score") if isinstance(result, dict) else None)
        or (result.get("relevance_score") if isinstance(result, dict) else None)
        or 0.0
    )
    metadata = (
        getattr(result, "metadata", None)
        or (result.get("metadata") if isinstance(result, dict) else None)
        or fallback_metadata
        or {}
    )
    return {
        "content": content,
        "score": float(score),
        "metadata": metadata,
        "source": "pageindex",
    }


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex nếu SDK khả dụng.

    Returns:
        List metadata manifest của documents đã xử lý/upload.
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if ".ocr_cache" in md_file.parts:
            continue
        documents.append(
            {
                "path": str(md_file),
                "relative_path": str(md_file.relative_to(STANDARDIZED_DIR)),
                "filename": md_file.name,
                "type": _document_type(md_file),
            }
        )

    uploaded = []
    pageindex_client = None
    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndex

            pageindex_client = PageIndex(api_key=PAGEINDEX_API_KEY)
        except Exception:
            pageindex_client = None

    for doc in documents:
        md_file = Path(doc["path"])
        metadata = {
            "source": doc["relative_path"],
            "filename": doc["filename"],
            "type": doc["type"],
        }
        status = "manifest_only"

        if pageindex_client is not None:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            try:
                if hasattr(pageindex_client, "upload"):
                    pageindex_client.upload(content=content, metadata=metadata)
                elif hasattr(pageindex_client, "add"):
                    pageindex_client.add(content=content, metadata=metadata)
                status = "uploaded"
            except Exception as exc:
                status = f"upload_failed: {type(exc).__name__}"

        uploaded.append({**metadata, "status": status})

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_MANIFEST_PATH.write_text(
        json.dumps(uploaded, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return uploaded


def _query_pageindex_sdk(query: str, top_k: int) -> list[dict]:
    if not PAGEINDEX_API_KEY:
        return []

    try:
        from pageindex import PageIndex

        client = PageIndex(api_key=PAGEINDEX_API_KEY)
    except Exception:
        return []

    try:
        if hasattr(client, "query"):
            raw_results = client.query(query=query, top_k=top_k)
        elif hasattr(client, "search"):
            raw_results = client.search(query=query, top_k=top_k)
        else:
            return []
    except Exception:
        return []

    return [_format_pageindex_result(result) for result in raw_results][:top_k]


def _local_pageindex_fallback(query: str, top_k: int) -> list[dict]:
    try:
        from src.task6_lexical_search import lexical_search

        results = lexical_search(query, top_k=top_k)
    except Exception:
        return []

    pageindex_results = []
    for result in results:
        pageindex_results.append(
            {
                "content": result.get("content", ""),
                "score": float(result.get("score", 0.0)),
                "metadata": {
                    **result.get("metadata", {}),
                    "pageindex_mode": "local_keyword_fallback",
                },
                "source": "pageindex",
            }
        )
    return pageindex_results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
    """
    query = query.strip()
    if not query or top_k <= 0:
        return []

    results = _query_pageindex_sdk(query, top_k=top_k)
    if not results:
        results = _local_pageindex_fallback(query, top_k=top_k)

    return results[:top_k]


if __name__ == "__main__":
    print("Preparing PageIndex documents...")
    manifest = upload_documents()
    print(f"Prepared {len(manifest)} documents")

    print("\nTest query:")
    for result in pageindex_search("hình phạt sử dụng ma túy", top_k=3):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
