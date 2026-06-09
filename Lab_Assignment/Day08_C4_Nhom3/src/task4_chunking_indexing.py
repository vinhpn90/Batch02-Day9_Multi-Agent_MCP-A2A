"""
Task 4 — SemanticChunker + local Vietnamese embedding + local vector index.

This implementation uses LangChain SemanticChunker backed by the local
Vietnamese SentenceTransformer model:

    dangvantuan/vietnamese-embedding

The model produces 768-dimensional vectors. The vector store is a local JSON
file, and the same embedding model/config can be reused by later retrieval
tasks.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
VECTOR_INDEX_PATH = INDEX_DIR / "semantic_chunks.json"


# =============================================================================
# CONFIGURATION
# =============================================================================

# SemanticChunker groups adjacent sentences by embedding similarity. A hard
# post-split cap is still applied because retrieval chunks should stay compact.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "semantic"

# Vietnamese embedding model requested by the user. It returns 768-d vectors.
EMBEDDING_MODEL = "dangvantuan/vietnamese-embedding"
EMBEDDING_DIM = 768
LOCAL_EMBEDDING_BATCH_SIZE = 16

# Local JSON index: stores chunks + dense vectors for downstream retrieval.
VECTOR_STORE = "local_json"

# SemanticChunker breakpoint config. Percentile 85 creates more focused chunks
# for mixed legal/news markdown than the default 95.
SEMANTIC_BREAKPOINT_TYPE = "percentile"
SEMANTIC_BREAKPOINT_AMOUNT = 85
MAX_SEMANTIC_SEGMENT_CHARS = 20_000


class SentenceTransformerEmbeddings:
    """Minimal LangChain Embeddings wrapper around SentenceTransformer."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, local_files_only=True)
        self.model.max_seq_length = 250

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=LOCAL_EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return embedding.tolist()


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_semantic_chunker():
    from langchain_experimental.text_splitter import SemanticChunker

    return SemanticChunker(
        embeddings=get_embedding_model(),
        breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_AMOUNT,
        sentence_split_regex=r"(?<=[.!?。！？])\s+|\n{2,}",
        min_chunk_size=120,
    )


def _normalize_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _hard_split(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if boundary > start + int(CHUNK_SIZE * 0.6):
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def _semantic_segments(text: str) -> list[str]:
    """Split very large markdown files before SemanticChunker to avoid OOM."""
    text = text.strip()
    if len(text) <= MAX_SEMANTIC_SEGMENT_CHARS:
        return [text] if text else []

    raw_parts = re.split(r"(?=^## Page \d+\b)", text, flags=re.MULTILINE)
    if len(raw_parts) == 1:
        raw_parts = re.split(r"\n\s*\n", text)

    segments = []
    current = ""
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        if len(part) > MAX_SEMANTIC_SEGMENT_CHARS:
            if current:
                segments.append(current)
                current = ""
            segments.extend(_hard_split(part))
            continue

        candidate = f"{current}\n\n{part}".strip() if current else part
        if len(candidate) > MAX_SEMANTIC_SEGMENT_CHARS and current:
            segments.append(current)
            current = part
        else:
            current = candidate

    if current:
        segments.append(current)
    return segments


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if ".ocr_cache" in md_file.parts:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        documents.append(
            {
                "content": _normalize_text(md_file.read_text(encoding="utf-8", errors="ignore")),
                "metadata": {
                    "source": str(relative_path),
                    "filename": md_file.name,
                    "type": doc_type,
                    "chunking_method": CHUNKING_METHOD,
                    "embedding_model": EMBEDDING_MODEL,
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng LangChain SemanticChunker.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunker = get_semantic_chunker()
    chunks = []

    for doc_index, doc in enumerate(documents):
        print(
            f"Chunking [{doc_index + 1}/{len(documents)}]: "
            f"{doc.get('metadata', {}).get('source', 'unknown')}",
            flush=True,
        )
        semantic_chunks = []
        segments = _semantic_segments(doc.get("content", ""))
        for segment_index, segment in enumerate(segments):
            if len(segments) > 1:
                print(f"  Segment {segment_index + 1}/{len(segments)}", flush=True)
            semantic_chunks.extend(chunker.split_text(segment))
        chunk_index = 0
        for semantic_chunk in semantic_chunks:
            for chunk_text in _hard_split(semantic_chunk):
                chunks.append(
                    {
                        "content": chunk_text,
                        "metadata": {
                            **doc.get("metadata", {}),
                            "doc_index": doc_index,
                            "chunk_index": chunk_index,
                            "chunk_size": len(chunk_text),
                            "semantic_breakpoint_type": SEMANTIC_BREAKPOINT_TYPE,
                            "semantic_breakpoint_amount": SEMANTIC_BREAKPOINT_AMOUNT,
                        },
                    }
                )
                chunk_index += 1

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng dangvantuan/vietnamese-embedding.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    embedding_model = get_embedding_model()
    embedded_chunks = []
    batch_size = LOCAL_EMBEDDING_BATCH_SIZE
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        print(f"Embedding chunks {start + 1}-{end}/{len(chunks)}", flush=True)
        texts = [chunk["content"] for chunk in chunks[start:end]]
        embeddings = embedding_model.embed_documents(texts)
        embedded_chunks.extend(
            {**chunk, "embedding": embedding}
            for chunk, embedding in zip(chunks[start:end], embeddings)
        )
    return embedded_chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào local JSON vector store.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "chunking_method": CHUNKING_METHOD,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "semantic_breakpoint_type": SEMANTIC_BREAKPOINT_TYPE,
            "semantic_breakpoint_amount": SEMANTIC_BREAKPOINT_AMOUNT,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "vector_store": VECTOR_STORE,
        },
        "chunks": chunks,
    }
    VECTOR_INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return VECTOR_INDEX_PATH


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_path = index_to_vectorstore(chunks)
    print(f"Indexed to vector store: {index_path}")


if __name__ == "__main__":
    run_pipeline()
