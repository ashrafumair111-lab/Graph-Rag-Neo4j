"""Chunking — split documents into overlapping chunks with stable chunk ids.

Every chunk gets a deterministic ``chunk_id`` (sha256 of ``source::index``)
which is used as the primary key in Qdrant and to link entities in Neo4j.
"""
from __future__ import annotations

import hashlib
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_logger, get_settings

logger = get_logger(__name__)


def _chunk_id(source: str, chunk_index: int) -> str:
    raw = f"{source}::{chunk_index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks, each tagged with ``chunk_id``."""
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )

    chunks: List[Document] = []
    for document in documents:
        parts = splitter.split_text(document.page_content)
        source = document.metadata.get("source", "unknown")
        for index, part in enumerate(parts):
            chunk = Document(
                page_content=part,
                metadata={
                    "source": source,
                    "chunk_index": index,
                    "chunk_id": _chunk_id(source, index),
                    "chunk_size": len(part),
                },
            )
            chunks.append(chunk)

    logger.info(
        "Chunked %d document(s) into %d chunk(s) "
        "(chunk_size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks