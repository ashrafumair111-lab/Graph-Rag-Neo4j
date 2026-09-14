"""Generate sample documents for quick testing of the Graph RAG pipeline.

Creates a ``data/`` folder (gitignored) with three entity-rich sample files
so you can exercise ingestion + retrieval without hunting for your own PDFs.

Usage:
    python create_sample_data.py
"""
from __future__ import annotations

from pathlib import Path

SAMPLES: dict[str, str] = {
    "tech_founders.txt": """Elon Musk founded Tesla in 2003 and SpaceX in 2002. Tesla is
headquartered in Austin, Texas, and produces electric vehicles such as the Model 3
and the Cybertruck. Elon Musk served as Chief Executive Officer of Tesla for many
years. In 2022, Elon Musk acquired Twitter and later renamed the platform to X.
SpaceX develops the Falcon 9 rocket and the Starship program from its headquarters
in Hawthorne, California.
""",
    "software_giants.txt": """Microsoft was founded by Bill Gates and Paul Allen in 1975.
Microsoft is headquartered in Redmond, Washington, and Satya Nadella is its Chief
Executive Officer. Microsoft develops the Windows operating system, the Azure cloud
platform, and the Office productivity suite. In 2023, Microsoft invested heavily in
OpenAI, the company behind ChatGPT.

Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
Apple is headquartered in Cupertino, California, and develops the iPhone and the Mac.

Google was founded by Larry Page and Sergey Brin in 1998. Google is headquartered
in Mountain View, California.
""",
    "ai_research.txt": """OpenAI is an artificial intelligence research organization founded
in 2015. Sam Altman is the Chief Executive Officer of OpenAI. OpenAI created the GPT
series of large language models, including GPT-4 and ChatGPT. OpenAI is headquartered
in San Francisco, California.

Anthropic was founded by former OpenAI executives and develops the Claude family of
models. Anthropic is also based in San Francisco, California.
""",
}


def main() -> None:
    target = Path(__file__).resolve().parent / "data"
    target.mkdir(parents=True, exist_ok=True)

    for name, content in SAMPLES.items():
        path = target / name
        path.write_text(content, encoding="utf-8")
        print(f"Created: {path}")

    print(f"\nDone. Now run:\n  python ingest.py {target} --recreate --max-chunks 6")


if __name__ == "__main__":
    main()