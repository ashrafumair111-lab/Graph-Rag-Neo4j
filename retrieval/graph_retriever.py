"""Graph retrieval — Neo4j neighbourhood facts around the query entities.

For every entity found in the graph we walk 1-2 hops over :code:`RELATED`
relationships and return each path as a human-readable "fact" together with the
chunk_ids that mention the involved entities (for cross-referencing with the
vector store and for citations).

If Neo4j is unreachable, every method logs the error and returns [].
"""
from __future__ import annotations

from typing import Any, Dict, List

from neo4j import GraphDatabase

from config import get_logger, get_settings

logger = get_logger(__name__)

_PRESENT_QUERY = """
UNWIND $names AS name
MATCH (e:Entity)
WHERE toLower(e.id) = toLower(name)
RETURN e.id AS id
"""

_FACTS_QUERY = """
UNWIND $names AS name
MATCH (e:Entity)
WHERE toLower(e.id) = toLower(name)
MATCH path = (e)-[:RELATED*1..2]-(other:Entity)
WITH path
ORDER BY length(path)
WITH path,
     [n IN nodes(path) | n.id] AS node_ids,
     [r IN relationships(path) | r.type] AS rel_types
RETURN DISTINCT node_ids, rel_types
LIMIT $limit
"""

_CHUNKS_QUERY = """
UNWIND $entity_ids AS entity_id
MATCH (e:Entity)
WHERE toLower(e.id) = toLower(entity_id)
OPTIONAL MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
RETURN toLower(e.id) AS entity, c.chunk_id AS chunk_id, c.source AS source
"""


def _fact_text(node_ids: List[str], rel_types: List[str]) -> str:
    """Render a path as text, e.g. 'Einstein -[WORKS_FOR]-> Princeton'."""
    parts = [node_ids[0]]
    for i, rel_type in enumerate(rel_types):
        parts.append(f"-[{rel_type}]->")
        parts.append(node_ids[i + 1])
    return " ".join(parts)


class GraphRetriever:
    """Fetches graph neighbourhood facts for a set of query entities."""

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

    # ------------------------------------------------------------------ query
    def _present_entities(self, names: List[str]) -> List[str]:
        with self._session() as session:
            records = session.run(
                _PRESENT_QUERY, names=[n for n in names if n]
            ).data()
        return [rec["id"] for rec in records]

    def _fetch_facts(self, names: List[str], hops: int, limit: int) -> List[Dict[str, Any]]:
        query = _FACTS_QUERY.replace("*1..2", f"*1..{hops}")
        with self._session() as session:
            records = session.run(query, names=names, limit=limit).data()
        return [{"node_ids": r["node_ids"], "rel_types": r["rel_types"]} for r in records]

    def _fetch_chunks(self, entity_ids: List[str]) -> Dict[str, List[str]]:
        """Return {lowercased entity id: [chunk_id, ...]}."""
        mapping: Dict[str, List[str]] = {}
        with self._session() as session:
            records = session.run(_CHUNKS_QUERY, entity_ids=entity_ids).data()
        for rec in records:
            entity = rec.get("entity")
            chunk_id = rec.get("chunk_id")
            if entity and chunk_id:
                mapping.setdefault(entity, []).append(chunk_id)
        return mapping

    # ------------------------------------------------------------------ entry
    def search(
        self,
        entities: List[str],
        hops: int = 2,
        max_facts: int = 60,
    ) -> List[Dict[str, Any]]:
        """Return graph facts for the query entities.

        Each fact: {text, type: "graph", source: "graph",
                    entities: [...], chunk_ids: [...], score: 0.0}
        """
        names = [name for name in (entities or []) if name and name.strip()]
        if not names:
            logger.info("Graph search skipped - no query entities.")
            return []

        try:
            present = self._present_entities(names)
            if not present:
                logger.info("No matching entities found in Neo4j for %s", names)
                return []

            raw_facts = self._fetch_facts(present, hops=hops, limit=max_facts)
            if not raw_facts:
                logger.info("No graph facts found around %s.", present)
                return []

            involved = sorted(
                {node for fact in raw_facts for node in fact["node_ids"]}
            )
            chunk_map = self._fetch_chunks(involved)

            results: List[Dict[str, Any]] = []
            for fact in raw_facts:
                node_ids, rel_types = fact["node_ids"], fact["rel_types"]
                chunk_ids = sorted(
                    {
                        cid
                        for node in node_ids
                        for cid in chunk_map.get(node.lower(), [])
                    }
                )
                results.append(
                    {
                        "text": _fact_text(node_ids, rel_types),
                        "type": "graph",
                        "source": "graph",
                        "entities": node_ids,
                        "chunk_ids": chunk_ids,
                        "score": 0.0,
                    }
                )

            logger.info(
                "Graph retriever returned %d fact(s) around %d entity(ies).",
                len(results),
                len(present),
            )
            return results

        except Exception as exc:  # noqa: BLE001 — graceful degradation
            logger.error("Graph retriever failed (Neo4j down?): %s", exc)
            return []