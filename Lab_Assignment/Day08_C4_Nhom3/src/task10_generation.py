"""
Task 10 — Generation Có Citation.

Pipeline:
    1. Retrieve top chunks từ Task 9
    2. Reorder context để giảm "lost in the middle"
    3. Format context với source labels
    4. Gọi LLM nếu có API key
    5. Fallback sang câu trả lời extractive có citation nếu API không sẵn sàng
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION
# =============================================================================

# Chọn 5 chunks: đủ evidence cho câu hỏi pháp luật/báo chí, không quá dài.
TOP_K = 5

# top_p=0.9 giữ câu trả lời tự nhiên nhưng vẫn tập trung vào context.
TOP_P = 0.9

# temperature=0.3 phù hợp RAG factual, giảm hallucination.
TEMPERATURE = 0.3

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()


SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    Giữ chunk quan trọng nhất ở đầu, đưa chunk quan trọng thứ hai về cuối,
    các chunk còn lại xen kẽ ở giữa.
    """
    if len(chunks) <= 2:
        return list(chunks)

    reordered = []
    for index in range(0, len(chunks), 2):
        reordered.append(chunks[index])

    start = len(chunks) - 1 if len(chunks) % 2 == 0 else len(chunks) - 2
    for index in range(start, 0, -2):
        reordered.append(chunks[index])

    return reordered


def _source_label(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source") or metadata.get("filename") or f"Source {index}"
    filename = Path(str(source)).name
    doc_type = metadata.get("type", "unknown")
    return f"{filename} | {doc_type}"


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.
    """
    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {index}")
        doc_type = metadata.get("type", "unknown")
        score = float(chunk.get("score", 0.0))
        content = chunk.get("content", "").strip()
        context_parts.append(
            f"[Document {index} | Source: {source} | Type: {doc_type} | Score: {score:.4f}]\n"
            f"{content}"
        )
    return "\n\n---\n\n".join(context_parts)


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _extractive_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    query_terms = set(re.findall(r"[\wÀ-ỹ]+", query.lower(), flags=re.UNICODE))
    answer_parts = []

    for index, chunk in enumerate(chunks[:3], start=1):
        sentences = _split_sentences(chunk.get("content", ""))
        if not sentences:
            continue

        best_sentence = max(
            sentences,
            key=lambda sentence: len(
                query_terms & set(re.findall(r"[\wÀ-ỹ]+", sentence.lower(), flags=re.UNICODE))
            ),
        )
        citation = _source_label(chunk, index)
        answer_parts.append(f"{best_sentence} [{citation}]")

    if not answer_parts:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    return " ".join(answer_parts)


def _call_openai_generation(user_message: str) -> str | None:
    # 1. Try OpenAI if key is valid (not empty and not sk-xxx)
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and not openai_key.startswith("sk-xxx") and openai_key != "sk-xxx":
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                timeout=25.0
            )
            return response.choices[0].message.content or None
        except Exception:
            pass

    # 2. Try Gemini via OpenAI compatibility (using GEMINI_API_KEY in .env)
    gemini_keys_str = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_keys_str:
        gemini_keys = [k.strip() for k in gemini_keys_str.split(",") if k.strip()]
        from openai import OpenAI
        for key in gemini_keys:
            try:
                # Use Gemini's OpenAI compatibility endpoint
                client = OpenAI(
                    api_key=key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
                response = client.chat.completions.create(
                    model="gemini-3.1-flash-lite",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    timeout=25.0
                )
                return response.choices[0].message.content or None
            except Exception as e:
                # Try next key if this one fails
                continue

    return None


def reformulate_query(query: str, chat_history: list[dict]) -> str:
    """
    Dựa trên lịch sử chat, viết lại query mới nhất thành câu độc lập đầy đủ ngữ cảnh để search.
    chat_history là danh sách các dict có dạng:
      {"role": "user"|"assistant", "content": str}
    """
    if not chat_history:
        return query

    # Format chat history
    history_str = ""
    for msg in chat_history[-6:]:  # Lấy tối đa 6 lượt chat gần nhất để tránh quá dài
        role_label = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role_label}: {msg['content']}\n"

    prompt = f"""Dưới đây là lịch sử cuộc trò chuyện giữa Người dùng (User) và Trợ lý (Assistant), cùng với câu hỏi mới nhất từ Người dùng.
Nhiệm vụ của bạn là phân tích ngữ cảnh cuộc trò chuyện và viết lại câu hỏi mới nhất thành một câu hỏi ĐỘC LẬP, ĐẦY ĐỦ Ý NGHĨA (bằng tiếng Việt) để có thể dùng tìm kiếm tài liệu chính xác nhất.

Yêu cầu:
1. Trả về DUY NHẤT câu hỏi được viết lại, không giải thích gì thêm, không thêm tag hay định dạng markdown khác.
2. Giữ nguyên ý định của người dùng, không tự tiện thêm thông tin không có trong lịch sử trò chuyện.
3. Nếu câu hỏi mới nhất đã đầy đủ ý nghĩa và không cần bổ sung ngữ cảnh, hãy trả về chính xác câu hỏi đó.

Lịch sử trò chuyện:
{history_str}
Câu hỏi mới nhất: {query}

Câu hỏi độc lập viết lại:"""

    # Gọi LLM để rewrite
    rewritten = _call_openai_generation(prompt)
    if rewritten:
        cleaned = rewritten.strip().strip('"').strip("'").strip()
        if cleaned:
            return cleaned
    return query


def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    score_threshold: float | None = None,
    use_reranking: bool = True,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str
        }
    """
    query = query.strip()
    if not query:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    retrieve_kwargs = {"top_k": top_k, "use_reranking": use_reranking}
    if score_threshold is not None:
        retrieve_kwargs["score_threshold"] = score_threshold

    chunks = retrieve(query, **retrieve_kwargs)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    answer = _call_openai_generation(user_message)
    if not answer:
        answer = _extractive_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
