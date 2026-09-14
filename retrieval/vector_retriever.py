"""Vector retrieval — Qdrant similarity search (top-k chunks).

Reuses :class:`ingestion.vector_store.QdrantVectorStore.search` so the Qdrant
client is created exactly once per process. Never raises: on any failure it
logs and returns an empty list (graceful degradation).
"""
from __future__ import annotations

from typing import Any, Dict, List

from config import get_logger
from ingestion.vector_store import QdrantVectorStore

logger = get_logger(__name__)


class VectorRetriever:
    """Retrieves the top-k most similar chunks from Qdrant for a query."""

    def __init__(self) -> None:
        self._store: QdrantVectorStore | None = None

    def _get_store(self) -> QdrantVectorStore:
        if self._store is None:
            self._store = QdrantVectorStore()
        return self._store

    def search(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Return up to top_k chunks: [{chunk_id, text, source, score}]."""
        if not query.strip():
            return []
        try:
            results = self._get_store().search(query, top_k=top_k)
            logger.info("Vector retriever returned %d chunk(s).", len(results))
            return results
        except Exception as exc:  # noqa: BLE001 — graceful degradation
            logger.error("Vector retriever failed: %s", exc)
            return []