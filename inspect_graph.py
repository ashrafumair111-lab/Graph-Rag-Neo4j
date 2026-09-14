"""Inspect what is currently stored in Neo4j and Qdrant.

Lets you verify ingestion results right from the terminal instead of
opening the Neo4j AuraDB browser.

Usage:
    python inspect_graph.py
"""
from __future__ import annotations

from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from config import get_logger, get_settings

logger = get_logger(__name__)


def inspect_neo4j(settings) -> None:
    print("\n" + "=" * 60)
    print("NEO4J")
    print("=" * 60)
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            connection_timeout=10,
        )
        with driver.session(database=settings.neo4j_database) as session:
            counts = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count "
                "ORDER BY label"
            ).data()
            if counts:
                for row in counts:
                    print(f"{row['label']:<10} {row['count']} node(s)")
            else:
                print("EMPTY GRAPH - pehle `python ingest.py <path>` chalao.")

            rels = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count "
                "ORDER BY type"
            ).data()
            if rels:
                for row in rels:
                    print(f"{row['type']:<16} {row['count']} edge(s)")

            entities = session.run(
                "MATCH (e:Entity) RETURN e.id AS id, e.type AS type LIMIT 15"
            ).data()
            print("\nSample entities:")
            for row in entities:
                print(f"  - {row['id']} ({row['type']})")

            facts = session.run(
                "MATCH (a:Entity)-[r:RELATED]->(b:Entity) "
                "RETURN a.id AS src, r.type AS rel, b.id AS tgt LIMIT 10"
            ).data()
            print("\nSample relationships:")
            for row in facts:
                print(f"  - {row['src']} -[{row['rel']}]-> {row['tgt']}")
        driver.close()
    except Exception as exc:  # noqa: BLE001
        logger.error("Neo4j inspect failed: %s", exc)


def inspect_qdrant(settings) -> None:
    print("\n" + "=" * 60)
    print("QDRANT")
    print("=" * 60)
    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=10,
        )
        if not client.collection_exists(settings.qdrant_collection):
            print(f"Collection '{settings.qdrant_collection}' does not exist yet.")
            return
        info = client.get_collection(settings.qdrant_collection)
        print(
            f"Collection: {settings.qdrant_collection} | "
            f"vectors={info.vectors_count} points={info.points_count} "
            f"status={info.status.name}"
        )

        sample = client.query_points(
            collection_name=settings.qdrant_collection,
            query=[0.0] * settings.cohere_embed_dim,
            limit=5,
            with_payload=True,
        )
        print("Sample points:")
        for point in sample.points:
            payload = point.payload or {}
            text = str(payload.get("text", ""))[:60]
            print(
                f"  - {payload.get('chunk_id')} | {payload.get('source')} | {text}..."
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("Qdrant inspect failed: %s", exc)


def main() -> None:
    settings = get_settings()
    print(
        f"Neo4j : {settings.neo4j_uri} (db: {settings.neo4j_database})\n"
        f"Qdrant: {settings.qdrant_url} / {settings.qdrant_collection}"
    )
    inspect_neo4j(settings)
    inspect_qdrant(settings)


if __name__ == "__main__":
    main()