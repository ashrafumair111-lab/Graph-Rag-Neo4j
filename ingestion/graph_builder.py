"""Graph building — extract entities/relationships from chunks and store in Neo4j.

Graph model (kept intentionally simple & queryable):

    (:Entity {id, type}) -[:RELATED {type}]-> (:Entity {id, type})
    (:Entity {id})-[:MENTIONED_IN]->(:Chunk {chunk_id, source, text})

Every entity is linked to the source chunk it was extracted from, enabling
cross-referencing between the vector store (chunk_id) and the graph.

Neo4j being unavailable never crashes the caller: methods log the error and
return an all-zero summary instead.
"""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from neo4j import GraphDatabase

from config import get_logger, get_settings

logger = get_logger(__name__)

# Small, consistent extraction vocabulary so graphs stay queryable.
ALLOWED_NODES = [
    "Person",
    "Organization",
    "Location",
    "Product",
    "Technology",
    "Concept",
    "Event",
]

ALLOWED_RELATIONSHIPS = [
    "FOUNDED",
    "FOUNDED_BY",
    "WORKS_FOR",
    "EMPLOYS",
    "CEO_OF",
    "PRESIDENT_OF",
    "HEADQUARTERED_IN",
    "LOCATED_IN",
    "PART_OF",
    "ACQUIRED",
    "DEVELOPS",
    "DEVELOPED",
    "CREATED",
    "CREATED_BY",
    "USES",
    "PROVIDES",
    "RELATED_TO",
    "MENTIONS",
]


def _norm_id(name: str) -> str:
    """Normalise an entity name to a stable id (collapse whitespace)."""
    return " ".join(str(name).replace("\u2019", "'").split())


class GraphBuilder:
    """Extracts a knowledge graph from chunks with an LLM and writes it to Neo4j."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._driver = None

    # ------------------------------------------------------------------ setup
    def _get_driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_username, self.settings.neo4j_password),
                connection_timeout=10,
            )
        return self._driver

    def _session(self):
        return self._get_driver().session(database=self.settings.neo4j_database)

    # ---------------------------------------------------------------- schema
    def ensure_schema(self) -> None:
        """Create uniqueness constraints / indexes if they do not exist."""
        with self._session() as session:
            session.run(
                "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            )
            session.run(
                "CREATE INDEX entity_type_index IF NOT EXISTS "
                "FOR (e:Entity) ON (e.type)"
            )
        logger.info("Neo4j schema (constraints & indexes) ensured.")

    # --------------------------------------------------------------- writing
    def _store_chunks(self, chunks: List[Document]) -> None:
        rows = [
            {
                "chunk_id": chunk.metadata["chunk_id"],
                "source": chunk.metadata.get("source", ""),
                "text": chunk.page_content,
            }
            for chunk in chunks
        ]
        with self._session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (c:Chunk {chunk_id: row.chunk_id})
                SET c.source = row.source, c.text = row.text
                """,
                rows=rows,
            )
        logger.info("Stored %d Chunk node(s) in Neo4j.", len(rows))

    @staticmethod
    def _run_batch(session, cypher: str, rows: List[Dict[str, Any]]) -> None:
        if rows:
            session.run(cypher, rows=rows)

    def _store_graph_rows(
        self,
        entity_rows: List[Dict[str, str]],
        mention_rows: List[Dict[str, str]],
        relate_rows: List[Dict[str, str]],
    ) -> Dict[str, int]:
        with self._session() as session:
            self._run_batch(
                session,
                """
                UNWIND $rows AS row
                MERGE (e:Entity {id: row.id})
                SET e.type = row.type
                """,
                entity_rows,
            )
            self._run_batch(
                session,
                """
                UNWIND $rows AS row
                MATCH (e:Entity {id: row.id})
                MATCH (c:Chunk {chunk_id: row.chunk_id})
                MERGE (e)-[:MENTIONED_IN]->(c)
                """,
                mention_rows,
            )
            self._run_batch(
                session,
                """
                UNWIND $rows AS row
                MATCH (a:Entity {id: row.source})
                MATCH (b:Entity {id: row.target})
                MERGE (a)-[r:RELATED {type: row.rel_type}]->(b)
                """,
                relate_rows,
            )
        return {
            "entities": len(entity_rows),
            "mentions": len(mention_rows),
            "relationships": len(relate_rows),
        }

    # ------------------------------------------------------------------ entry
    def build_and_store(self, chunks: List[Document]) -> Dict[str, Any]:
        """Run LLM extraction over chunks and write everything to Neo4j."""
        if not chunks:
            logger.warning("No chunks provided to graph builder - nothing to do.")
            return {
                "graph_documents": 0,
                "entities": 0,
                "mentions": 0,
                "relationships": 0,
            }

        settings = self.settings

        # 1) LLM-based entity/relationship extraction.
        llm = ChatGroq(
            model=settings.groq_model,
            groq_api_key=settings.groq_api_key,
            temperature=0.0,
        )
        transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=ALLOWED_NODES,
            allowed_relationships=ALLOWED_RELATIONSHIPS,
            strict_mode=True,
        )
        logger.info("Running LLM graph extraction on %d chunk(s) ...", len(chunks))
        graph_documents = transformer.convert_to_graph_documents(chunks)
        logger.info(
            "Extracted %d graph document(s) from %d chunk(s).",
            len(graph_documents),
            len(chunks),
        )

        # 2) Store chunk nodes so MENTIONED_IN links can be created.
        self.ensure_schema()
        self._store_chunks(chunks)

        # 3) Accumulate entity / mention / relationship rows (deduped).
        entity_rows: List[Dict[str, str]] = []
        mention_rows: List[Dict[str, str]] = []
        relate_rows: List[Dict[str, str]] = []

        for graph_doc in graph_documents:
            chunk_id = None
            if graph_doc.source is not None:
                chunk_id = (graph_doc.source.metadata or {}).get("chunk_id")

            for node in graph_doc.nodes:
                nid = _norm_id(node.id)
                if nid:
                    entity_rows.append({"id": nid, "type": node.type or "Entity"})
            for rel in graph_doc.relationships:
                src = _norm_id(rel.source.id)
                tgt = _norm_id(rel.target.id)
                if src and tgt:
                    entity_rows.append({"id": src, "type": rel.source.type or "Entity"})
                    entity_rows.append({"id": tgt, "type": rel.target.type or "Entity"})
                    relate_rows.append(
                        {"source": src, "target": tgt, "rel_type": rel.type or "RELATED"}
                    )
            if chunk_id:
                for node in graph_doc.nodes:
                    nid = _norm_id(node.id)
                    if nid:
                        mention_rows.append({"id": nid, "chunk_id": chunk_id})

        # Dedupe rows before batching.
        entity_rows = list({r["id"]: r for r in entity_rows}.values())
        mention_rows = list(
            {(r["id"], r["chunk_id"]): r for r in mention_rows}.values()
        )
        relate_rows = list(
            {(r["source"], r["rel_type"], r["target"]): r for r in relate_rows}.values()
        )

        summary = self._store_graph_rows(entity_rows, mention_rows, relate_rows)
        summary["graph_documents"] = len(graph_documents)
        logger.info(
            "Stored %(entities)d entity nodes, %(mentions)d MENTIONED_IN links, "
            "%(relationships)d relationships in Neo4j.",
            summary,
        )
        return summary