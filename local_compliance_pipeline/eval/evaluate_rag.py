"""Compare local RAG chunking and retrieval configurations.

Run after Ollama is ready:
    uv run python eval/evaluate_rag.py

The input labels are in ``eval/rag_queries.jsonl``. Add real compliance
questions and expected PDF sources/sections before treating results as final.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from dotenv import load_dotenv
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "backend" / "data"
QUERIES = Path(__file__).with_name("rag_queries.jsonl")
load_dotenv(ROOT / ".env")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def fixed_chunks(source: str, page: int, text: str) -> list[dict]:
    return [{"text": text[i:i + 800].strip(), "source": source, "page": page}
            for i in range(0, len(text), 650) if text[i:i + 800].strip()]


def sentence_chunks(source: str, page: int, text: str) -> list[dict]:
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) > 800 and current:
            chunks.append({"text": current.strip(), "source": source, "page": page})
            current = current[-150:] + " " + sentence
        else:
            current += " " + sentence
    if current.strip():
        chunks.append({"text": current.strip(), "source": source, "page": page})
    return chunks


def load_chunks(kind: str) -> list[dict]:
    chunker = fixed_chunks if kind == "fixed" else sentence_chunks
    chunks = []
    for pdf in DATA_DIR.glob("*.pdf"):
        for page_no, page in enumerate(PdfReader(str(pdf)).pages, start=1):
            chunks.extend(chunker(pdf.name, page_no, page.extract_text() or ""))
    return chunks


class BM25:
    def __init__(self, documents: list[str]):
        self.docs = [Counter(tokens(doc)) for doc in documents]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.average = sum(self.lengths) / max(len(self.lengths), 1)
        self.df = Counter(term for doc in self.docs for term in doc)

    def rank(self, query: str, limit: int) -> list[int]:
        scores = defaultdict(float)
        for term in tokens(query):
            idf = math.log(1 + (len(self.docs) - self.df[term] + 0.5) / (self.df[term] + 0.5))
            for index, document in enumerate(self.docs):
                tf = document[term]
                scores[index] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self.lengths[index] / max(self.average, 1)))
        return sorted(range(len(self.docs)), key=lambda index: scores[index], reverse=True)[:limit]


def hyde(query: str, base_url: str, model: str) -> str:
    prompt = f"Write a short hypothetical compliance-policy passage that answers this question. Do not explain: {query}"
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}).encode()
    request = Request(f"{base_url.rstrip('/')}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read())["response"]


def evaluate(chunker: str, method: str, rerank: bool, queries: list[dict], top_k: int) -> dict:
    chunks = load_chunks(chunker)
    texts = [chunk["text"] for chunk in chunks]
    bm25 = BM25(texts)
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    embed_model = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    chat_model = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:7b-instruct")
    client = chromadb.EphemeralClient()
    collection = client.create_collection("rag_eval", embedding_function=OllamaEmbeddingFunction(url=base_url, model_name=embed_model))
    collection.add(ids=[str(i) for i in range(len(chunks))], documents=texts, metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks])
    hits, reciprocal_ranks, latencies = 0, [], []
    for item in queries:
        started = time.perf_counter()
        query = hyde(item["query"], base_url, chat_model) if method == "hyde" else item["query"]
        dense_ids = [int(i) for i in collection.query(query_texts=[query], n_results=min(20, len(chunks)))["ids"][0]]
        lexical_ids = bm25.rank(item["query"], 20)
        if method == "hybrid":
            scores = defaultdict(float)
            for ranking in (dense_ids, lexical_ids):
                for rank, index in enumerate(ranking, start=1): scores[index] += 1 / (60 + rank)
            ranked = sorted(scores, key=scores.get, reverse=True)
        else:
            ranked = dense_ids
        if rerank:
            candidates = set(ranked[:20])
            ranked = sorted(candidates, key=lambda index: (index in lexical_ids, -lexical_ids.index(index) if index in lexical_ids else -999), reverse=True)
        ranked = ranked[:top_k]
        expected = set(item["expected_sources"])
        matching = [rank for rank, index in enumerate(ranked, start=1) if chunks[index]["source"] in expected]
        hits += bool(matching); reciprocal_ranks.append(1 / matching[0] if matching else 0); latencies.append(time.perf_counter() - started)
    return {"chunker": chunker, "method": method, "rerank": rerank, "chunks": len(chunks), "recall_at_k": hits / len(queries), "mrr_at_k": sum(reciprocal_ranks) / len(queries), "mean_latency_seconds": sum(latencies) / len(latencies)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunkers", default="fixed,sentence")
    parser.add_argument("--methods", default="dense,hybrid,hyde")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="eval/rag_results.json")
    args = parser.parse_args()
    queries = [json.loads(line) for line in QUERIES.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = [evaluate(chunker, method, rerank, queries, args.top_k)
               for chunker in args.chunkers.split(",") for method in args.methods.split(",") for rerank in (False, True)]
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results: print(result)


if __name__ == "__main__": main()
