"""Reranking — narrows the combined context down to RAG_TOP_N using Cohere Rerank.

If the Cohere API fails, we fall back to the pre-rerank ordering (top_n items
sorted by their original scores) so the pipeline never breaks.
"""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_cohere import CohereRerank

from config import get_logger, get_settings

logger = get_logger(__name__)


class Reranker:
    """Cohere reranker over a mixed list of vector chunks and graph facts."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._reranker: CohereRerank | None = None

    def _get_reranker(self) -> CohereRerank:
        if self._reranker is None:
            self._reranker = CohereRerank(
                model=self.settings.cohere_rerank_model,
                cohere_api_key=self.settings.cohere_api_key,
            )
        return self._reranker

    def rerank(
        self,
        query: str,
        items: List[Dict[str, Any]],
        top_n: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Re-order ``items`` by relevance to ``query`` and keep the top-n.

        Items must be dicts containing a ``text`` key; the original item dicts
        are returned (with ``score`` updated to the rerank relevance score).
        """
        if not items:
            return []

        top_n = min(top_n or self.settings.top_n, len(items))

        try:
            documents = [item["text"] for item in items]
            results = self._get_reranker().rerank(
                documents=documents,
                query=query,
                top_n=top_n,
            )
            ranked: List[Dict[str, Any]] = []
            for result in results:
                item = dict(items[result["index"]])
                item["score"] = float(result["relevance_score"])
                ranked.append(item)
            logger.info("Reranker returned %d item(s) (top_n=%d).", len(ranked), top_n)
            return ranked
        except Exception as exc:  # noqa: BLE001 — graceful degradation
            logger.warning(
                "Rerank failed (%s) — using pre-rerank ordering instead.", exc
            )
            ordered = sorted(
                items, key=lambda item: item.get("score", 0.0), reverse=True
            )
            return ordered[:top_n]