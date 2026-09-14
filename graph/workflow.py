"""LangGraph state machine for the Graph RAG query pipeline.

Flow::

    query_input
        -> entity_extraction
            -> vector_search  \\   (executed in parallel)
            -> graph_search   /
        -> merge_context
        -> rerank
        -> generate_answer
        -> END

Retrieval never crashes: if the graph DB or the vector store is unavailable the
corresponding node simply returns an empty result (graceful degradation).
"""
from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Any, Dict, List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import get_logger, get_settings
from retrieval.graph_retriever import GraphRetriever
from retrieval.reranker import Reranker
from retrieval.vector_retriever import VectorRetriever

logger = get_logger(__name__)

# ---------------------------------------------------------------------- state


class GraphRAGState(TypedDict, total=False):
    query: str
    entities: List[Dict[str, str]]
    vector_results: List[Dict[str, Any]]
    graph_results: List[Dict[str, Any]]
    combined_context: List[Dict[str, Any]]
    reranked_context: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]
    warnings: List[str]


# --------------------------------------------------------------------- models


def _get_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        model=settings.groq_model,
        groq_api_key=settings.groq_api_key,
        temperature=0.0,
    )


class _QueryEntity(BaseModel):
    name: str = Field(description="Canonical entity name as it appears in the query")
    type: str = Field(
        description="Entity type, e.g. Person, Organization, Location, Product"
    )


class _EntityList(BaseModel):
    entities: List[_QueryEntity]


_EXTRACTION_PROMPT = """You extract the important named entities from a user question.
Only include entities that could exist in a knowledge graph
(people, organizations, locations, products, technologies, concepts, events).
If there are no meaningful entities, return an empty list."""


def _naive_entities(query: str) -> List[Dict[str, str]]:
    """Fallback extraction: capitalized phrases when structured output fails."""
    matches = re.findall(
        r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)*\b", query
    )
    seen = set()
    entities = []
    for match in matches:
        match = match.strip()
        if match and len(match) >= 2 and match.lower() not in seen:
            seen.add(match.lower())
            entities.append({"name": match, "type": "Entity"})
    return entities[:10]


# ---------------------------------------------------------------------- nodes


def query_input(state: GraphRAGState) -> Dict[str, Any]:
    """Normalise the incoming query."""
    query = (state.get("query") or "").strip()
    logger.info("query_input: %s", query)
    return {"query": query}


def entity_extraction(state: GraphRAGState) -> Dict[str, Any]:
    """Identify important entities from the query using the Groq LLM."""
    query = state.get("query") or ""
    if not query.strip():
        return {"entities": [], "warnings": ["Empty query."]}

    start = time.perf_counter()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _EXTRACTION_PROMPT),
            ("human", "Question: {query}"),
        ]
    )
    entities: List[Dict[str, str]] = []

    try:
        chain = prompt | _get_llm().with_structured_output(_EntityList)
        result = chain.invoke({"query": query})
        for entry in (getattr(result, "entities", None) or []):
            name = getattr(entry, "name", None) or (
                entry.get("name") if isinstance(entry, dict) else None
            )
            etype = getattr(entry, "type", None) or (
                entry.get("type") if isinstance(entry, dict) else None
            )
            if name and str(name).strip():
                entities.append(
                    {"name": str(name).strip(), "type": str(etype or "Entity")}
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Structured extraction failed (%s) — falling back to naive extraction.",
            exc,
        )
        entities = _naive_entities(query)

    logger.info(
        "entity_extraction: %d entity(ies) in %.2fs.",
        len(entities),
        time.perf_counter() - start,
    )
    return {"entities": entities}


def vector_search(state: GraphRAGState) -> Dict[str, Any]:
    """Top-k similarity search against Qdrant (runs in parallel with graph search)."""
    query = state.get("query") or ""
    start = time.perf_counter()
    results = VectorRetriever().search(query, top_k=get_settings().top_k)
    logger.info(
        "vector_search: %d chunk(s) in %.2fs.",
        len(results),
        time.perf_counter() - start,
    )
    return {"vector_results": results}


def graph_search(state: GraphRAGState) -> Dict[str, Any]:
    """1-2 hop neighbourhood facts from Neo4j around the query entities."""
    entities = state.get("entities") or []
    names = [entity.get("name", "") for entity in entities if entity.get("name")]
    start = time.perf_counter()
    results = GraphRetriever().search(names, hops=2)
    logger.info(
        "graph_search: %d fact(s) in %.2fs.",
        len(results),
        time.perf_counter() - start,
    )
    return {"graph_results": results}


def merge_context(state: GraphRAGState) -> Dict[str, Any]:
    """Combine vector chunks and graph facts into one deduplicated context list."""
    combined: List[Dict[str, Any]] = []
    seen_ids = set()

    for item in state.get("vector_results") or []:
        chunk_id = item.get("chunk_id")
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        combined.append(
            {
                "id": chunk_id,
                "type": "vector",
                "text": item.get("text", ""),
                "source": item.get("source", ""),
                "score": float(item.get("score", 0.0)),
                "chunk_ids": [chunk_id] if chunk_id else [],
            }
        )

    seen_texts = set()
    graph_count = 0
    for item in state.get("graph_results") or []:
        text = item.get("text", "")
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        graph_count += 1
        combined.append(
            {
                "id": f"graph:{abs(hash(text))}",
                "type": "graph",
                "text": text,
                "source": item.get("source", "graph"),
                "score": 0.0,
                "chunk_ids": item.get("chunk_ids") or [],
            }
        )

    logger.info(
        "merge_context: %d item(s) total (%d vector, %d graph).",
        len(combined),
        len(combined) - graph_count,
        graph_count,
    )
    return {"combined_context": combined}


def rerank(state: GraphRAGState) -> Dict[str, Any]:
    """Narrow the combined context to RAG_TOP_N with Cohere rerank."""
    query = state.get("query") or ""
    combined = state.get("combined_context") or []
    start = time.perf_counter()
    ranked = Reranker().rerank(query, combined, top_n=get_settings().top_n)
    logger.info(
        "rerank: %d item(s) after reranking in %.2fs.",
        len(ranked),
        time.perf_counter() - start,
    )
    return {"reranked_context": ranked}


_ANSWER_SYSTEM = """You are a precise retrieval-augmented assistant.
Answer the user's question using ONLY the numbered context snippets below.

Rules:
- Support every factual claim with one or more citation markers, e.g. [1] or [2][4].
- The markers must reference snippet numbers from the CONTEXT section.
- If the context does not contain the answer, say exactly:
  "I could not find enough information in the provided context."
- Do not guess or use outside knowledge. Be concise and accurate."""


def _format_context(items: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, item in enumerate(items, start=1):
        kind = "vector-chunk" if item.get("type") == "vector" else "graph-fact"
        header = f"[{i}] ({kind})"
        if item.get("source"):
            header += f" | source: {item['source']}"
        blocks.append(f"{header}\n{item.get('text', '')}")
    return "\n\n".join(blocks)


def generate_answer(state: GraphRAGState) -> Dict[str, Any]:
    """Generate the final answer with citations from the reranked context."""
    query = state.get("query") or ""
    context_items = state.get("reranked_context") or []

    if not context_items:
        message = (
            "No relevant context found. Make sure documents have been ingested "
            "first (python ingest.py <path>), then retry."
        )
        logger.warning("generate_answer: empty context.")
        return {"answer": message, "citations": []}

    context_block = _format_context(context_items)
    messages = [
        ("system", _ANSWER_SYSTEM),
        ("human", f"CONTEXT:\n{context_block}\n\nQUESTION: {query}\n\nANSWER:"),
    ]

    start = time.perf_counter()
    response = _get_llm().invoke(messages)
    answer = str(response.content or "")

    cited_indexes = sorted(
        {
            int(m)
            for m in re.findall(r"\[(\d+)\]", answer)
            if m.isdigit() and 1 <= int(m) <= len(context_items)
        }
    )
    citations: List[Dict[str, Any]] = []
    for index in cited_indexes:
        item = context_items[index - 1]
        citations.append(
            {
                "index": index,
                "type": item.get("type"),
                "source": item.get("source"),
                "chunk_id": (item.get("chunk_ids") or [item.get("id")])[:3],
                "snippet": item.get("text", "")[:200],
            }
        )

    logger.info(
        "generate_answer: %d chars, %d citation(s) in %.2fs.",
        len(answer),
        len(citations),
        time.perf_counter() - start,
    )
    return {"answer": answer, "citations": citations}


# ---------------------------------------------------------------- graph build


def build_workflow():
    """Compile the LangGraph pipeline."""
    workflow = StateGraph(GraphRAGState)

    workflow.add_node("query_input", query_input)
    workflow.add_node("entity_extraction", entity_extraction)
    workflow.add_node("vector_search", vector_search)
    workflow.add_node("graph_search", graph_search)
    workflow.add_node("merge_context", merge_context)
    workflow.add_node("rerank", rerank)
    workflow.add_node("generate_answer", generate_answer)

    workflow.add_edge(START, "query_input")
    workflow.add_edge("query_input", "entity_extraction")

    # Parallel retrieval fan-out.
    workflow.add_edge("entity_extraction", "vector_search")
    workflow.add_edge("entity_extraction", "graph_search")

    # Both retrievers must finish before merging.
    workflow.add_edge("vector_search", "merge_context")
    workflow.add_edge("graph_search", "merge_context")

    workflow.add_edge("merge_context", "rerank")
    workflow.add_edge("rerank", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


@lru_cache(maxsize=1)
def get_workflow():
    """Build the workflow once and reuse the compiled graph."""
    return build_workflow()


def answer_query(query: str) -> Dict[str, Any]:
    """Convenience helper: run a single query through the full pipeline."""
    return get_workflow().invoke({"query": query})