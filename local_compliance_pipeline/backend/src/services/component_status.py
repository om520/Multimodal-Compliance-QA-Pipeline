"""Read-only health checks for local models and the vector database."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from backend.src.services.video_indexer import get_model_load_states


def _state(status: str, detail: str, **extra: object) -> dict:
    return {"status": status, "detail": detail, **extra}


def _project_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def check_ollama() -> dict:
    """Check that Ollama is reachable and configured models are installed."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b-instruct")
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    try:
        with urlopen(f"{base_url}/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        return _state(
            "offline",
            f"Ollama is not reachable at {base_url}: {error}",
            chat_model=chat_model,
            embedding_model=embedding_model,
        )

    available = {model.get("name") for model in payload.get("models", [])}
    missing = [model for model in (chat_model, embedding_model) if model not in available]
    if missing:
        return _state(
            "incomplete",
            f"Ollama is running, but these models are missing: {', '.join(missing)}",
            chat_model=chat_model,
            embedding_model=embedding_model,
            available_models=sorted(name for name in available if name),
        )

    return _state(
        "ready",
        "Ollama and both configured models are available.",
        chat_model=chat_model,
        embedding_model=embedding_model,
    )


def check_chroma(project_root: Path) -> dict:
    """Check that the expected local Chroma collection exists and has data."""
    persist_dir = _project_path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"), project_root)
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "compliance_rules")
    if not persist_dir.is_dir():
        return _state(
            "missing",
            "ChromaDB has not been indexed. Run backend/scripts/index_documents.py.",
            path=str(persist_dir),
            collection=collection_name,
        )

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(persist_dir))
        collection = client.get_collection(collection_name)
        count = collection.count()
    except Exception as error:
        return _state(
            "error",
            f"Could not open the configured Chroma collection: {error}",
            path=str(persist_dir),
            collection=collection_name,
        )

    status = "ready" if count else "empty"
    detail = "Chroma collection is ready." if count else "Chroma collection exists but contains no documents."
    return _state(status, detail, path=str(persist_dir), collection=collection_name, documents=count)


def component_status(project_root: Path) -> dict:
    """Return all component states without forcing heavyweight model loads."""
    ollama = check_ollama()
    chroma = check_chroma(project_root)
    media_models = get_model_load_states()
    ready = ollama["status"] == "ready" and chroma["status"] == "ready"

    return {
        "overall": "ready" if ready else "not_ready",
        "api": _state("ready", "FastAPI is serving this response."),
        "ollama": ollama,
        "chromadb": chroma,
        "media_models": media_models,
    }
