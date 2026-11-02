"""
index_documents.py – Ingest compliance PDF documents into ChromaDB (local vector store).

Replaces Azure AI Search + Azure OpenAI Embeddings with:
  - ChromaDB  (persistent local vector store)
  - Ollama    (local embeddings via nomic-embed-text)

Reads all PDFs from ``backend/data/``, splits them into chunks, generates
embeddings, and upserts them into the configured ChromaDB collection.

Usage:
    python backend/scripts/index_documents.py
"""

import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Allow running from the repo root: python backend/scripts/index_documents.py
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

load_dotenv()

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("indexer")

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "compliance_rules")


# ── PDF loading ───────────────────────────────────────────────────────────────

def load_pdfs(data_dir: Path):
    """Load all PDFs from data_dir, return list of (text, source_filename) tuples."""
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", data_dir)
        return []

    docs = []
    for pdf_path in pdf_files:
        logger.info("Loading PDF: %s", pdf_path.name)
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        docs.append((text, pdf_path.name))
        logger.info("  → %d chars extracted", len(text))
    return docs


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks. Returns list of (chunk_text, metadata) tuples."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, {"source": source, "start_char": start}))
        start += chunk_size - overlap
    return chunks


# ── Main indexing ─────────────────────────────────────────────────────────────

def main():
    logger.info("=== Compliance Document Indexer (Local / ChromaDB) ===")
    logger.info("Data directory  : %s", DATA_DIR)
    logger.info("ChromaDB path   : %s", CHROMA_PERSIST_DIR)
    logger.info("Collection      : %s", CHROMA_COLLECTION_NAME)
    logger.info("Embedding model : %s via %s", OLLAMA_EMBEDDING_MODEL, OLLAMA_BASE_URL)

    # ── Load PDFs ──────────────────────────────────────────────────────────────
    raw_docs = load_pdfs(DATA_DIR)
    if not raw_docs:
        logger.error("No documents to index. Add PDFs to %s", DATA_DIR)
        sys.exit(1)

    # ── Chunk documents ────────────────────────────────────────────────────────
    all_chunks = []
    for text, source in raw_docs:
        chunks = chunk_text(text, source)
        logger.info("  %s → %d chunks", source, len(chunks))
        all_chunks.extend(chunks)

    logger.info("Total chunks to index: %d", len(all_chunks))

    # ── Connect to ChromaDB ────────────────────────────────────────────────────
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    ef = OllamaEmbeddingFunction(
        url=OLLAMA_BASE_URL,
        model_name=OLLAMA_EMBEDDING_MODEL,
    )

    # Recreate collection (full re-index)
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        logger.info("Deleted existing collection '%s'", CHROMA_COLLECTION_NAME)
    except Exception:
        pass  # Collection didn't exist yet

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Upsert in batches ─────────────────────────────────────────────────────
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [c[0] for c in batch]
        metadatas = [c[1] for c in batch]
        ids = [str(uuid.uuid4()) for _ in batch]

        collection.add(documents=texts, metadatas=metadatas, ids=ids)
        logger.info("  Indexed batch %d/%d", min(i + batch_size, len(all_chunks)), len(all_chunks))

    logger.info("✅  Indexing complete — %d chunks in ChromaDB collection '%s'",
                len(all_chunks), CHROMA_COLLECTION_NAME)


if __name__ == "__main__":
    main()
