"""Project configuration.

All settings are loaded from environment variables defined in the `.env` file
at the project root. Nothing is hardcoded here. This module is case-insensitive
w.r.t. variable names (e.g. `groq_api_key` and `GROQ_API_KEY` both work).

Use ``from config import settings`` or ``get_settings()`` anywhere in the codebase.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Project root = directory containing this file.
ROOT_DIR = Path(__file__).resolve().parent

# Load variables from the .env file at the project root
# (does not override variables already present in the environment).
load_dotenv(ROOT_DIR / ".env")

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a configured, idempotent logger writing to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _get(name: str, default: str = "") -> str:
    """Fetch an env var trying exact, lowercase, then uppercase spellings."""
    value = os.getenv(name)
    if value is None:
        value = os.getenv(name.lower())
    if value is None:
        value = os.getenv(name.upper())
    if value is None:
        return default
    return value.strip()


def _get_int(name: str, default: int) -> int:
    try:
        return int(_get(name) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable settings snapshot used across the whole project."""

    # --- LLM (Groq) ---
    groq_api_key: str
    groq_model: str
    # --- Cohere (embeddings + rerank) ---
    cohere_api_key: str
    cohere_embed_model: str
    cohere_rerank_model: str
    cohere_embed_dim: int
    # --- Qdrant (vector store) ---
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str
    # --- Neo4j (graph store) ---
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    # --- RAG knobs ---
    chunk_size: int
    chunk_overlap: int
    top_k: int
    top_n: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        groq_api_key=_get("groq_api_key"),
        groq_model=_get("groq_model", "openai/gpt-oss-120b"),
        cohere_api_key=_get("cohere_api_key"),
        cohere_embed_model=_get("cohere_embed_model", "embed-english-v3.0"),
        cohere_rerank_model=_get("cohere_rerank_model", "rerank-english-v3.0"),
        cohere_embed_dim=_get_int("cohere_embed_dim", 1024),
        qdrant_url=_get("qdrant_url"),
        qdrant_api_key=_get("qdrant_api_key"),
        qdrant_collection=_get("qdrant_collection", "graph_rag_chunks"),
        neo4j_uri=_get("neo4j_uri"),
        neo4j_username=_get("neo4j_username", "neo4j"),
        neo4j_password=_get("neo4j_password"),
        neo4j_database=_get("neo4j_database", "neo4j"),
        chunk_size=_get_int("rag_chunk_size", 500),
        chunk_overlap=_get_int("rag_overlap", 80),
        top_k=_get_int("rag_top_k", 20),
        top_n=_get_int("rag_top_n", 5),
    )


# Module-level singleton.
settings = get_settings()


def require_keys(*keys: str) -> List[str]:
    """Return the list of keys that are currently empty/missing.

    Use ``missing = require_keys("groq_api_key", "cohere_api_key")`` at startup
    to give a friendly early error instead of a cryptic API failure.
    """
    return [key for key in keys if not _get(key)]