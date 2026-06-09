import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# We must import our retrieval and generation functions
from src.task10_generation import generate_with_citation, reformulate_query
from src.task9_retrieval_pipeline import SCORE_THRESHOLD, DEFAULT_TOP_K
from src.supervisor_workers_agent import run_supervisor_agent

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot - Pháp Luật & Tin Tức Ma Tuý",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI Styling Custom CSS
st.markdown("""
<style>
    /* Gradient Main Title */
    .main-title {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.8rem;
        font-weight: 400;
    }
    
    /* Custom status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .status-active {
        background-color: #D1FAE5;
        color: #065F46;
    }
    
    .status-inactive {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    
    /* Source box highlighting */
    .source-box {
        background-color: #F9FAFB;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1.2rem;
        margin: 0.6rem 0;
        border-radius: 0.375rem;
        font-size: 0.9rem;
    }
    
    .source-header {
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 0.3rem;
    }
    
    .source-meta {
        font-size: 0.8rem;
        color: #6B7280;
        margin-bottom: 0.4rem;
    }
    
    .source-content {
        color: #374151;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Description
st.markdown('<div class="main-title">⚖️ Hệ Thống Hỏi Đáp Pháp Luật Phòng Chống Ma Tuý</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">RAG Chatbot kết hợp Tìm kiếm ngữ nghĩa (Semantic), Tra cứu từ khóa (Lexical), Reranking và Fallback PageIndex</div>', unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar configuration and status
with st.sidebar:
    st.header("⚙️ Cấu Hình RAG")
    
    st.subheader("Tham số Tìm kiếm")
    top_k = st.slider("Số lượng tài liệu (Top K)", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    score_threshold = st.slider("Ngưỡng điểm (Threshold)", min_value=0.0, max_value=1.0, value=SCORE_THRESHOLD, step=0.05)
    use_reranking = st.toggle("Sử dụng Reranking (Jina/CrossEncoder)", value=True)
    use_supervisor_agent = st.toggle("Supervisor - Workers Agent", value=True)

    if use_supervisor_agent:
        st.markdown(
            '<span class="status-badge status-active">● Pattern: Supervisor + 3 Workers</span>',
            unsafe_allow_html=True
        )
        st.caption("Workers: Query Understanding, Retrieval, Answer Generation")
    else:
        st.markdown(
            '<span class="status-badge status-inactive">● Pattern: Classic RAG Pipeline</span>',
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    st.subheader("🔍 Trạng Thái Hệ Thống")
    
    # Check Index Exist
    index_path = "data/index/semantic_chunks.json"
    if os.path.exists(index_path):
        st.markdown('<span class="status-badge status-active">● Vector Index: Sẵn sàng</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-inactive">● Vector Index: Thiếu</span>', unsafe_allow_html=True)
        
    # Check LLM Key
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    
    if openai_key and not openai_key.startswith("sk-xxx"):
        st.markdown('<span class="status-badge status-active">● OpenAI API: Sẵn sàng</span>', unsafe_allow_html=True)
    elif gemini_key:
        st.markdown('<span class="status-badge status-active">● Gemini API: Sẵn sàng (Fallback)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-inactive">● LLM API: Trích xuất extractive</span>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Action buttons
    if st.button("🗑️ Xóa Lịch Sử Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if assistant message contains them
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            ret_src = message.get("retrieval_source", "hybrid")
            badge_color = "green" if ret_src == "hybrid" else "orange"
            
            with st.expander(f"🔍 Tài liệu tham khảo ({len(message['sources'])} chunks - nguồn: :{badge_color}[{ret_src}])"):
                for idx, src in enumerate(message["sources"], 1):
                    meta = src.get("metadata", {})
                    filename = meta.get("filename") or meta.get("source") or "Không rõ nguồn"
                    doc_type = meta.get("type", "unknown")
                    score = src.get("score", 0.0)
                    
                    st.markdown(
                        f'<div class="source-box">'
                        f'  <div class="source-header">{idx}. {filename} <span class="source-card-score">Score: {score:.4f}</span></div>'
                        f'  <div class="source-meta">Loại tài liệu: <b>{doc_type}</b></div>'
                        f'  <div class="source-content">"{src.get("content", "").strip()}"</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        if message["role"] == "assistant" and message.get("agent_trace"):
            with st.expander("🧭 Supervisor - Workers trace"):
                for step in message["agent_trace"]:
                    st.markdown(
                        f"- **{step['worker']}** ({step['elapsed_ms']} ms): {step['summary']}"
                    )

# Chat Input
query = st.chat_input("Nhập câu hỏi của bạn ở đây (ví dụ: Tội tàng trữ ma tuý bị phạt tù bao nhiêu năm?)...")

if query:
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Context reformulation for follow-up questions
    history = st.session_state.messages[:-1]
    search_query = query
    if history and not use_supervisor_agent:
        with st.spinner("Đang phân tích ngữ cảnh cuộc hội thoại..."):
            search_query = reformulate_query(query, history)
            if search_query != query:
                st.info(f"🔄 Câu hỏi được tối ưu hóa ngữ cảnh: *\"{search_query}\"*")
                
    # Call RAG Generation pipeline
    with st.spinner("Đang tìm kiếm thông tin và tạo câu trả lời..."):
        if use_supervisor_agent:
            result = run_supervisor_agent(
                query=query,
                chat_history=history,
                top_k=top_k,
                score_threshold=score_threshold,
                use_reranking=use_reranking
            )
            search_query = result.get("search_query", query)
            if search_query != query:
                st.info(f"🔄 Query Worker đã viết lại câu hỏi: *\"{search_query}\"*")
        else:
            result = generate_with_citation(
                query=search_query,
                top_k=top_k,
                score_threshold=score_threshold,
                use_reranking=use_reranking
            )
        
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        
        # Display sources
        sources = result.get("sources", [])
        ret_src = result.get("retrieval_source", "none")
        if sources:
            badge_color = "green" if ret_src == "hybrid" else "orange"
            with st.expander(f"🔍 Tài liệu tham khảo ({len(sources)} chunks - nguồn: :{badge_color}[{ret_src}])"):
                for idx, src in enumerate(sources, 1):
                    meta = src.get("metadata", {})
                    filename = meta.get("filename") or meta.get("source") or "Không rõ nguồn"
                    doc_type = meta.get("type", "unknown")
                    score = src.get("score", 0.0)
                    
                    st.markdown(
                        f'<div class="source-box">'
                        f'  <div class="source-header">{idx}. {filename} <span class="source-card-score">Score: {score:.4f}</span></div>'
                        f'  <div class="source-meta">Loại tài liệu: <b>{doc_type}</b></div>'
                        f'  <div class="source-content">"{src.get("content", "").strip()}"</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        agent_trace = result.get("trace", [])
        if agent_trace:
            with st.expander("🧭 Supervisor - Workers trace"):
                st.markdown(
                    f"**Supervisor elapsed:** {result.get('total_elapsed_ms', 0):.2f} ms"
                )
                for step in agent_trace:
                    st.markdown(
                        f"- **{step['worker']}** ({step['elapsed_ms']} ms): {step['summary']}"
                    )
                    
    # Save assistant response to memory
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": sources,
        "retrieval_source": ret_src,
        "agent_trace": result.get("trace", []),
    })
