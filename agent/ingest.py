"""
P14a — ZORC RAG Ingestion
Loads PDFs from data/papers/, splits into chunks, embeds with
sentence-transformers/all-MiniLM-L6-v2, and persists to ChromaDB.

Usage:
    conda activate zorc_pipeline
    python agent/ingest.py
"""

import argparse
import json
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "pbody_literature"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_pdfs(papers_dir: Path) -> list:
    """Load all PDFs from papers_dir, return list of Document objects."""
    docs = []
    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[ingest] No PDFs found in {papers_dir}", file=sys.stderr)
        sys.exit(1)

    for pdf_path in pdf_files:
        print(f"[ingest] Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        # Attach clean filename as source metadata
        for page in pages:
            page.metadata["source"] = pdf_path.name
            page.metadata["paper"] = pdf_path.stem
        docs.extend(pages)
        print(f"         → {len(pages)} pages")

    print(f"\n[ingest] Total pages loaded: {len(docs)}")
    return docs


def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Total chunks after splitting: {len(chunks)}")
    return chunks


def build_vectorstore(chunks: list) -> Chroma:
    print(f"[ingest] Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"[ingest] Building ChromaDB at: {CHROMA_DIR}")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"[ingest] ChromaDB populated: {vectorstore._collection.count()} vectors")
    return vectorstore


def save_manifest(chunks: list) -> None:
    """Save a small JSON manifest summarising what was indexed."""
    sources = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    manifest = {
        "total_chunks": len(chunks),
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "collection": COLLECTION_NAME,
        "sources": sources,
    }
    manifest_path = Path(__file__).resolve().parent / "ingest_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[ingest] Manifest saved: {manifest_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Ingest P-body PDFs into ChromaDB")
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=PAPERS_DIR,
        help="Directory containing PDF files (default: data/papers/)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing ChromaDB and rebuild from scratch",
    )
    args = parser.parse_args()

    if args.rebuild and CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print(f"[ingest] Removed existing ChromaDB at {CHROMA_DIR}")

    docs = load_pdfs(args.papers_dir)
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    save_manifest(chunks)
    print("\n[ingest] Done. Run agent/rag_query.py to query the knowledge base.")


if __name__ == "__main__":
    main()
