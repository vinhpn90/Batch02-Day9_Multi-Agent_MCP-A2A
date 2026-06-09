"""
Supervisor - Workers agent for Day08 RAG chatbot.

Pattern:
    Supervisor
      -> Query Worker: normalize/reformulate the user question
      -> Retrieval Worker: run hybrid retrieval + fallback
      -> Answer Worker: generate answer with citations

The public function `run_supervisor_agent()` returns the same core shape as
`generate_with_citation()` so the Streamlit app can switch between the classic
pipeline and the agentic pipeline without changing the UI contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .task10_generation import (
    _call_openai_generation,
    _extractive_answer,
    format_context,
    reformulate_query,
    reorder_for_llm,
)
from .task9_retrieval_pipeline import DEFAULT_TOP_K, SCORE_THRESHOLD, retrieve


@dataclass
class WorkerResult:
    """Structured result returned by each worker."""

    name: str
    summary: str
    elapsed_ms: float
    payload: dict[str, Any]


class QueryWorker:
    """Worker 1: prepare a standalone retrieval query."""

    name = "query_worker"

    def run(self, query: str, chat_history: list[dict]) -> WorkerResult:
        started = perf_counter()
        search_query = reformulate_query(query, chat_history) if chat_history else query
        changed = search_query.strip() != query.strip()
        elapsed_ms = (perf_counter() - started) * 1000
        return WorkerResult(
            name=self.name,
            summary=(
                "Reformulated follow-up question for retrieval."
                if changed
                else "Question is already standalone; no rewrite needed."
            ),
            elapsed_ms=elapsed_ms,
            payload={
                "original_query": query,
                "search_query": search_query,
                "changed": changed,
            },
        )


class RetrievalWorker:
    """Worker 2: retrieve and prepare evidence chunks."""

    name = "retrieval_worker"

    def run(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = SCORE_THRESHOLD,
        use_reranking: bool = True,
    ) -> WorkerResult:
        started = perf_counter()
        chunks = retrieve(
            query,
            top_k=top_k,
            score_threshold=score_threshold,
            use_reranking=use_reranking,
        )
        reordered = reorder_for_llm(chunks)
        elapsed_ms = (perf_counter() - started) * 1000
        source = chunks[0].get("source", "none") if chunks else "none"
        return WorkerResult(
            name=self.name,
            summary=f"Retrieved {len(chunks)} chunks via {source}.",
            elapsed_ms=elapsed_ms,
            payload={
                "chunks": chunks,
                "reordered_chunks": reordered,
                "retrieval_source": source,
            },
        )


class AnswerWorker:
    """Worker 3: produce grounded answer with citations."""

    name = "answer_worker"

    def run(self, query: str, chunks: list[dict]) -> WorkerResult:
        started = perf_counter()
        context = format_context(chunks)
        user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
        answer = _call_openai_generation(user_message)
        generation_mode = "llm"
        if not answer:
            answer = _extractive_answer(query, chunks)
            generation_mode = "extractive"

        elapsed_ms = (perf_counter() - started) * 1000
        return WorkerResult(
            name=self.name,
            summary=f"Generated final answer using {generation_mode} mode.",
            elapsed_ms=elapsed_ms,
            payload={
                "answer": answer,
                "generation_mode": generation_mode,
            },
        )


class RagSupervisor:
    """Supervisor that coordinates the three Day08 RAG workers."""

    def __init__(self) -> None:
        self.query_worker = QueryWorker()
        self.retrieval_worker = RetrievalWorker()
        self.answer_worker = AnswerWorker()

    def run(
        self,
        query: str,
        chat_history: list[dict] | None = None,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = SCORE_THRESHOLD,
        use_reranking: bool = True,
    ) -> dict:
        started = perf_counter()
        history = chat_history or []
        trace: list[dict] = []

        query_result = self.query_worker.run(query=query, chat_history=history)
        trace.append(_trace_item(query_result))
        search_query = query_result.payload["search_query"]

        retrieval_result = self.retrieval_worker.run(
            query=search_query,
            top_k=top_k,
            score_threshold=score_threshold,
            use_reranking=use_reranking,
        )
        trace.append(_trace_item(retrieval_result))
        reordered_chunks = retrieval_result.payload["reordered_chunks"]

        answer_result = self.answer_worker.run(
            query=search_query,
            chunks=reordered_chunks,
        )
        trace.append(_trace_item(answer_result))

        total_elapsed_ms = (perf_counter() - started) * 1000
        return {
            "answer": answer_result.payload["answer"],
            "sources": retrieval_result.payload["chunks"],
            "retrieval_source": retrieval_result.payload["retrieval_source"],
            "search_query": search_query,
            "agent_pattern": "supervisor_workers",
            "workers": ["query_worker", "retrieval_worker", "answer_worker"],
            "trace": trace,
            "total_elapsed_ms": total_elapsed_ms,
        }


def _trace_item(result: WorkerResult) -> dict:
    return {
        "worker": result.name,
        "summary": result.summary,
        "elapsed_ms": round(result.elapsed_ms, 2),
    }


def run_supervisor_agent(
    query: str,
    chat_history: list[dict] | None = None,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> dict:
    """Run the Supervisor - Workers RAG agent."""
    supervisor = RagSupervisor()
    return supervisor.run(
        query=query,
        chat_history=chat_history,
        top_k=top_k,
        score_threshold=score_threshold,
        use_reranking=use_reranking,
    )
