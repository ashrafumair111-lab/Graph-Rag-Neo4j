"""Standalone ingestion script.

Loads documents from a file or folder, chunks them, embeds them into Qdrant,
and extracts/stores a knowledge graph in Neo4j.

Examples:
    python ingest.py ./data                     # ingest everything
    python ingest.py C:\\docs\\notes.txt --recreate
    python ingest.py ./data --skip-graph        # vector-only ingestion
    python ingest.py ./data --max-chunks 10     # quick test with 10 chunks
"""
from __future__ import annotations

import argparse
import time

from config import get_logger, require_keys
from ingestion.chunker import chunk_documents
from ingestion.document_loader import load_documents
from ingestion.graph_builder import GraphBuilder
from ingestion.vector_store import QdrantVectorStore

logger = get_logger(__name__)


def run_ingestion(args: argparse.Namespace) -> None:
    total_start = time.perf_counter()

    missing = require_keys(
        "groq_api_key",
        "cohere_api_key",
        "qdrant_url",
        "qdrant_api_key",
        "neo4j_uri",
    )
    if missing:
        logger.warning(
            "Missing/empty env keys: %s — some steps will fail. Check your .env.",
            ", ".join(missing),
        )

    # 1) Load documents -----------------------------------------------------
    step_start = time.perf_counter()
    try:
        documents = load_documents(args.path)
    except FileNotFoundError as exc:
        logger.error(
            "Could not find input path. %s "
            "(Run `python create_sample_data.py` to generate sample docs.)",
            exc,
        )
        return
    if not documents:
        logger.error("No documents loaded from %s — aborting.", args.path)
        return
    logger.info(
        "Step [load] finished in %.2fs (%d document(s)).",
        time.perf_counter() - step_start,
        len(documents),
    )

    # 2) Chunk ---------------------------------------------------------------
    step_start = time.perf_counter()
    chunks = chunk_documents(documents)
    if args.max_chunks and len(chunks) > args.max_chunks:
        logger.warning(
            "--max-chunks: keeping only first %d of %d chunks.",
            args.max_chunks,
            len(chunks),
        )
        chunks = chunks[: args.max_chunks]
    logger.info(
        "Step [chunk] finished in %.2fs (%d chunk(s)).",
        time.perf_counter() - step_start,
        len(chunks),
    )

    # 3) Vector store (Qdrant) ----------------------------------------------
    vector_count = 0
    if args.skip_vector:
        logger.warning("Skipping vector store (--skip-vector).")
    else:
        step_start = time.perf_counter()
        store = QdrantVectorStore()
        store.ensure_collection(force_recreate=args.recreate)
        vector_count = store.add_documents(chunks)
        logger.info(
            "Step [vector] finished in %.2fs (%d point(s)).",
            time.perf_counter() - step_start,
            vector_count,
        )

    # 4) Graph store (Neo4j) --------------------------------------------------
    graph_summary = {}
    if args.skip_graph:
        logger.warning("Skipping graph store (--skip-graph).")
    else:
        step_start = time.perf_counter()
        builder = GraphBuilder()
        try:
            graph_summary = builder.build_and_store(chunks)
            logger.info(
                "Step [graph] finished in %.2fs.",
                time.perf_counter() - step_start,
            )
        except Exception as exc:  # noqa: BLE001 — never crash the whole script
            logger.error("Graph building failed (%s). Vector store is still ready.", exc)

    total = time.perf_counter() - total_start
    logger.info(
        "Ingestion complete in %.2fs. vectors=%d graph=%s",
        total,
        vector_count,
        graph_summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into Qdrant (vectors) and Neo4j (graph)."
    )
    parser.add_argument(
        "path",
        help="File or folder containing .pdf/.txt/.md/.docx documents.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop & recreate the Qdrant collection before inserting.",
    )
    parser.add_argument(
        "--skip-vector",
        action="store_true",
        help="Skip embedding + Qdrant storage.",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip LLM graph extraction + Neo4j storage.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Process only the first N chunks (useful for testing).",
    )
    args = parser.parse_args()
    run_ingestion(args)


if __name__ == "__main__":
    main()