"""
index_documents.py â€“ Ingest compliance PDF documents into Azure AI Search.

Reads all PDFs from ``backend/data/``, splits them into chunks, generates
embeddings via Azure OpenAI, and upserts them into the configured Azure AI
Search vector index.

Usage:
    python backend/scripts/index_documents.py
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow running from the repo root: python backend/scripts/index_documents.py
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from langchain_community.vectorstores import AzureSearch  # type: ignore
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("indexer")

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_pdfs(data_dir: Path):
    """Load all PDFs from data_dir and return a flat list of LangChain Documents."""
    all_docs = []
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", data_dir)
        return all_docs

    for pdf_path in pdf_files:
        logger.info("Loading: %s", pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        # Tag each chunk with its source filename
        for doc in docs:
            doc.metadata["source_file"] = pdf_path.name
        all_docs.extend(docs)
        logger.info("  â†’ Loaded %d pages from %s", len(docs), pdf_path.name)

    return all_docs


def split_documents(documents):
    """Split documents into overlapping chunks for semantic search."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d pages into %d chunks (size=%d, overlap=%d).",
                len(documents), len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI__EMBEDDING_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    )


def index_documents(chunks) -> None:
    """Upsert document chunks into Azure AI Search."""
    embeddings = get_embeddings()

    logger.info("Connecting to Azure AI Search index: %s", os.environ["AZURE_SEARCH_INDEX_NAME"])
    vector_store = AzureSearch(
        azure_search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        azure_search_key=os.environ["AZURE_SEARCH_API_KEY"],
        index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
        embedding_function=embeddings.embed_query,
    )

    # Batch upsert in groups of 100 to avoid request size limits
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vector_store.add_documents(batch)
        logger.info("  Indexed batch %d/%d (%d chunks).", i // batch_size + 1,
                    -(-len(chunks) // batch_size), len(batch))

    logger.info("âœ…  Indexing complete â€” %d chunks indexed.", len(chunks))


def main() -> None:
    logger.info("=== Compliance Document Indexer ===")
    logger.info("Data directory: %s", DATA_DIR)

    docs = load_pdfs(DATA_DIR)
    if not docs:
        logger.error("No documents loaded. Exiting.")
        sys.exit(1)

    chunks = split_documents(docs)
    index_documents(chunks)


if __name__ == "__main__":
    main()
