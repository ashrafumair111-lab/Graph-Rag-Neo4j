"""CLI entry point for the Graph RAG system.

Examples:
    python main.py                      # interactive Q&A loop
    python main.py -q "Who founded X?"  # single question, then exit
"""
from __future__ import annotations

import argparse

from config import get_logger, require_keys
from graph.workflow import get_workflow

logger = get_logger(__name__)

_EXIT_COMMANDS = {"exit", "quit", "bye", "q"}


def print_result(result) -> None:
    """Pretty-print the workflow result (answer + citations)."""
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.get("answer", ""))

    citations = result.get("citations") or []
    if citations:
        print("\nCITATIONS")
        print("-" * 70)
        for citation in citations:
            source = citation.get("source") or "?"
            kind = citation.get("type") or "?"
            chunk_id = citation.get("chunk_id") or []
            chunk_str = ", ".join(str(c) for c in chunk_id)
            label = f"[{citation.get('index')}] ({kind}) source={source}"
            if chunk_str:
                label += f" chunk={chunk_str}"
            print(label)
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions to the Graph RAG system (LangChain + LangGraph)."
    )
    parser.add_argument(
        "-q",
        "--query",
        help="Ask a single question and exit (skips the interactive loop).",
    )
    args = parser.parse_args()

    missing = require_keys(
        "groq_api_key", "cohere_api_key", "qdrant_url", "qdrant_api_key"
    )
    if missing:
        logger.warning(
            "Missing/empty env keys: %s — check your .env file.",
            ", ".join(missing),
        )

    workflow = get_workflow()

    if args.query:
        result = workflow.invoke({"query": args.query})
        print_result(result)
        return

    print("Graph RAG CLI — type your question below ('exit' to quit).")
    while True:
        try:
            query = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in _EXIT_COMMANDS:
            print("Bye!")
            break

        result = workflow.invoke({"query": query})
        print_result(result)


if __name__ == "__main__":
    main()