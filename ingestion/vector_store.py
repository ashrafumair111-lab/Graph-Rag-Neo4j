"""Vector store — embed chunks with Cohere and store them in Qdrant Cloud.

Payload stored per point:
    chunk_id, source, chunk_index, text
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import get_logger, get_settings

logger = get_logger(__name__)

_UPSERT_BATCH = 64  # safe upsert batch size for Qdrant


def _point_id(chunk_id: str) -> str:
    """Deterministic UUID for a chunk_id (Qdrant point ids must be UUID/int).

    Same chunk_id always maps to the same UUID, so re-running ingest simply
    replaces the existing point instead of duplicating it.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_id)))


class QdrantVectorStore:
    """Thin wrapper around the Qdrant cloud client for chunk embeddings."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: QdrantClient | None = None
        self._embeddings: CohereEmbeddings | None = None

    # ------------------------------------------------------------------ utils
    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                timeout=30,
            )
        return self._client

    def _get_embeddings(self) -> CohereEmbeddings:
        if self._embeddings is None:
            self._embeddings = CohereEmbeddings(
                model=self.settings.cohere_embed_model,
                cohere_api_key=self.settings.cohere_api_key,
            )
        return self._embeddings

    # ---------------------------------------------------------------- writing
    def ensure_collection(self, force_recreate: bool = False) -> None:
        """Create the collection if missing (optionally drop & recreate it)."""
        client = self._get_client()
        name = self.settings.qdrant_collection
        exists = client.collection_exists(name)

        if exists and force_recreate:
            logger.warning("Dropping and recreating collection '%s' ...", name)
            client.delete_collection(name)
            exists = False

        if not exists:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.settings.cohere_embed_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d, metric=cosine)",
                name,
                self.settings.cohere_embed_dim,
            )
        else:
            logger.info("Qdrant collection '%s' already exists.", name)

    def add_documents(self, documents: List[Document]) -> int:
        """Embed and upsert chunks into Qdrant. Returns number of points stored."""
        if not documents:
            return 0

        embeddings = self._get_embeddings()
        client = self._get_client()
        name = self.settings.qdrant_collection

        texts = [doc.page_content for doc in documents]
        total = 0

        for start in range(0, len(texts), _UPSERT_BATCH):
            batch_texts = texts[start : start + _UPSERT_BATCH]
            batch_docs = documents[start : start + _UPSERT_BATCH]

            vectors = embeddings.embed_documents(batch_texts)

            points = [
                PointStruct(
                    id=_point_id(doc.metadata["chunk_id"]),
                    vector=vector,
                    payload={
                        "chunk_id": doc.metadata["chunk_id"],
                        "source": doc.metadata.get("source", ""),
                        "chunk_index": doc.metadata.get("chunk_index", -1),
                        "text": doc.page_content,
                    },
                )
                for doc, vector in zip(batch_docs, vectors)
            ]

            client.upsert(collection_name=name, points=points, wait=True)
            total += len(points)

        logger.info("Stored %d chunk vector(s) in Qdrant collection '%s'.", total, name)
        return total

    # ----------------------------------------------------------------- reading
    def search(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Return the top-k most similar chunks for a query (cosine similarity).

        Returns a list of dicts: {chunk_id, text, source, score}.
        Never raises — on any failure it logs the error and returns [].
        """
        top_k = top_k or self.settings.top_k
        results: List[Dict[str, Any]] = []

        try:
            query_vector = self._get_embeddings().embed_query(query)
            response = self._get_client().query_points(
                collection_name=self.settings.qdrant_collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

            for point in response.points:
                payload = point.payload or {}
                results.append(
                    {
                        "chunk_id": payload.get("chunk_id"),
                        "text": payload.get("text", ""),
                        "source": payload.get("source", ""),
                        "score": float(point.score),
                    }
                )
        except Exception as exc:  # noqa: BLE001 — graceful degradation
            logger.error("Vector search failed (falling back to empty): %s", exc)

        logger.info(
            "Vector search returned %d result(s) for top_k=%d.",
            len(results),
            top_k,
        )
        return results