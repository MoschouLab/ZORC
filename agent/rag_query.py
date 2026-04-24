"""
P14a — ZORC RAG Query Interface
Loads the ChromaDB vector store built by agent/ingest.py and exposes
query_literature(question, k) for semantic retrieval over P-body papers.

Usage (CLI):
    conda activate zorc_pipeline
    python agent/rag_query.py "What RNA features predict P-body enrichment?"

Usage (Python):
    from agent.rag_query import query_literature
    results = query_literature("How does heat stress affect P-body assembly?")
    for r in results:
        print(r["source"], r["page"], r["text"][:200])
"""

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Any

AGENT_DIR = Path(__file__).resolve().parent
CHROMA_DIR = AGENT_DIR / "chroma_db"
COLLECTION_NAME = "pbody_literature"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K = 5

# Module-level cache so repeated calls don't reload the model
_vectorstore = None


def _load_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_DIR}. "
            "Run `python agent/ingest.py` first."
        )

    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    _vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    return _vectorstore


def query_literature(question: str, k: int = DEFAULT_K) -> list[dict[str, Any]]:
    """
    Retrieve the top-k most relevant chunks for *question*.

    Returns a list of dicts:
        {
          "rank":   int,          # 1-based relevance rank
          "score":  float,        # cosine similarity (higher = more relevant)
          "paper":  str,          # PDF stem (filename without .pdf)
          "source": str,          # full PDF filename
          "page":   int,          # 0-based page number from PyPDFLoader
          "text":   str,          # raw chunk text
        }
    """
    vs = _load_vectorstore()
    results = vs.similarity_search_with_relevance_scores(question, k=k)

    hits = []
    for rank, (doc, score) in enumerate(results, start=1):
        hits.append(
            {
                "rank": rank,
                "score": round(score, 4),
                "paper": doc.metadata.get("paper", "unknown"),
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", -1),
                "text": doc.page_content,
            }
        )
    return hits


def _pretty_print(question: str, hits: list[dict]) -> None:
    width = 80
    print("=" * width)
    print(f"QUERY: {question}")
    print("=" * width)
    for h in hits:
        print(
            f"\n[{h['rank']}] {h['source']}  |  page {h['page'] + 1}"
            f"  |  score={h['score']:.4f}"
        )
        print("-" * width)
        wrapped = textwrap.fill(h["text"], width=width)
        print(wrapped)
    print("\n" + "=" * width)


def main():
    parser = argparse.ArgumentParser(
        description="Query P-body literature ChromaDB via semantic search"
    )
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument(
        "-k", type=int, default=DEFAULT_K, help="Number of results (default: 5)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the three standard P14a demo queries",
    )
    args = parser.parse_args()

    demo_queries = [
        "What RNA features predict P-body enrichment?",
        "How does heat stress affect P-body assembly?",
        "What is the role of DCP1 in mRNA decapping?",
    ]

    if args.demo:
        for q in demo_queries:
            hits = query_literature(q, k=args.k)
            _pretty_print(q, hits)
    elif args.question:
        hits = query_literature(args.question, k=args.k)
        _pretty_print(args.question, hits)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
