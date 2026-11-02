"""
retrieval_and_reranking.py – Advanced Retrieval & Re-ranking Suite for Compliance RAG.

Implements:
  1. Dense Retrieval (ChromaDB + nomic-embed-text)
  2. BM25 Sparse Keyword Retrieval (Pure Python Okapi BM25)
  3. Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
  4. LLM Cross-Encoder Re-Ranking (Qwen 2.5 7B)
  5. Format-Aware Metadata Filtered Hybrid Retrieval (Context-Aware)
"""

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:7b-instruct")
CHROMA_DIR = str(ROOT_DIR / "chroma_db")


# ── Text Preprocessing for BM25 ──────────────────────────────────────────────

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into clean words."""
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class BM25Retriever:
    """Pure-Python Okapi BM25 implementation for lexical keyword search."""

    def __init__(self, documents: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.corpus_size = len(documents)
        self.doc_tokens = [tokenize(d["text"] + " " + d.get("title", "") + " " + d.get("category", "")) for d in documents]
        self.doc_lens = [len(dt) for dt in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / max(1, self.corpus_size)

        # Compute term frequencies and document frequencies
        self.doc_freqs: Counter = Counter()
        self.term_freqs: List[Counter] = []
        for dt in self.doc_tokens:
            tf = Counter(dt)
            self.term_freqs.append(tf)
            for token in tf.keys():
                self.doc_freqs[token] += 1

    def score(self, query: str) -> List[Tuple[Dict[str, Any], float]]:
        q_tokens = tokenize(query)
        scores = []

        for idx, doc in enumerate(self.documents):
            score = 0.0
            doc_len = self.doc_lens[idx]
            tf_counter = self.term_freqs[idx]

            for q in q_tokens:
                if q not in tf_counter:
                    continue
                tf = tf_counter[q]
                df = self.doc_freqs[q]
                # Standard Okapi IDF
                idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * ((tf * (self.k1 + 1.0)) / denom)

            scores.append((doc, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class ComplianceRetrievalSuite:
    """Unified engine implementing Dense, Sparse, Hybrid, and Re-ranking retrieval."""

    def __init__(self):
        # 1. Connect to ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.ef = OllamaEmbeddingFunction(url=OLLAMA_BASE_URL, model_name=EMBED_MODEL)
        self.collection = self.chroma_client.get_collection("compliance_rules", embedding_function=self.ef)

        # 2. Fetch all rule documents into memory for sparse & re-ranking pipelines
        raw = self.collection.get(include=["documents", "metadatas"])
        self.all_rules = []
        for rid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"]):
            self.all_rules.append({
                "rule_id": rid,
                "text": doc,
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
                "source": meta.get("source", ""),
                "severity": meta.get("severity", ""),
                "applies_to": meta.get("applies_to", "")
            })

        # 3. Initialize BM25 engine
        self.bm25 = BM25Retriever(self.all_rules)

        # 4. Lazy-loaded ChatOllama for Re-ranking / HyDE
        self._llm = None
        self._cross_tok = None
        self._cross_mod = None

    @property
    def llm(self) -> ChatOllama:
        if self._llm is None:
            self._llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=CHAT_MODEL, temperature=0.0)
        return self._llm

    @property
    def cross_encoder(self):
        if self._cross_mod is None:
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self._cross_tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self._cross_mod = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
            self._cross_mod.eval()
        return self._cross_tok, self._cross_mod

    # ── Method 1: Dense Semantic Retrieval ───────────────────────────────────
    def retrieve_dense(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        retrieved = []
        if results and results.get("ids") and results["ids"][0]:
            for rid, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                retrieved.append({
                    "rule_id": rid,
                    "title": meta.get("title", ""),
                    "category": meta.get("category", ""),
                    "severity": meta.get("severity", ""),
                    "source": meta.get("source", ""),
                    "score": round(1.0 - float(dist), 4),
                    "text": doc
                })
        return retrieved

    # ── Method 2: BM25 Sparse Keyword Retrieval ──────────────────────────────
    def retrieve_sparse(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        scored_docs = self.bm25.score(query)
        retrieved = []
        for doc, score in scored_docs[:top_k]:
            retrieved.append({
                "rule_id": doc["rule_id"],
                "title": doc.get("title", ""),
                "category": doc.get("category", ""),
                "severity": doc.get("severity", ""),
                "source": doc.get("source", ""),
                "score": round(float(score), 4),
                "text": doc["text"]
            })
        return retrieved

    # ── Method 3: Hybrid Retrieval with RRF ──────────────────────────────────
    def retrieve_hybrid_rrf(self, query: str, top_k: int = 6, rrf_k: int = 60) -> List[Dict[str, Any]]:
        # Fetch top-12 from dense and sparse
        dense_results = self.retrieve_dense(query, top_k=12)
        sparse_results = self.retrieve_sparse(query, top_k=12)

        rrf_scores: Dict[str, float] = {}
        rule_map: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results, 1):
            rid = item["rule_id"]
            rule_map[rid] = item
            rrf_scores[rid] = rrf_scores.get(rid, 0.0) + (1.0 / (rrf_k + rank))

        for rank, item in enumerate(sparse_results, 1):
            rid = item["rule_id"]
            if rid not in rule_map:
                rule_map[rid] = item
            rrf_scores[rid] = rrf_scores.get(rid, 0.0) + (1.0 / (rrf_k + rank))

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        retrieved = []
        for rid, score in sorted_items[:top_k]:
            item = dict(rule_map[rid])
            item["score"] = round(score, 6)
            retrieved.append(item)
        return retrieved

    # ── Method 4: LLM Cross-Encoder Re-Ranking ───────────────────────────────
    def retrieve_with_reranking(self, query: str, sample_meta: Dict[str, Any], top_k: int = 6) -> List[Dict[str, Any]]:
        # 1. Start with Top-8 candidates from Hybrid RRF
        candidates = self.retrieve_hybrid_rrf(query, top_k=8)

        # 2. Re-rank prompt for Qwen 2.5 7B
        prompt_lines = [
            f"Video Context: Title: '{sample_meta.get('title')}', Format: '{sample_meta.get('ad_format')}', Duration: {sample_meta.get('duration_seconds')}s.",
            f"Query Extracted from Video: {query[:350]}",
            "\nEvaluate the relevance of each candidate compliance rule. Score each candidate from 0 to 10 on whether the video asset may potentially violate or be regulated by this rule.",
            "Candidates:"
        ]
        for idx, c in enumerate(candidates, 1):
            prompt_lines.append(f"[{idx}] ID: {c['rule_id']} | Title: {c['title']} | Rule Summary: {c['text'][:150]}")

        prompt_lines.append("\nReturn ONLY a JSON array of objects with 'id' and integer 'relevance' (0-10), sorted highest to lowest.")
        prompt_str = "\n".join(prompt_lines)

        try:
            resp = self.llm.invoke([
                SystemMessage(content="You are a strict compliance search ranker. Output ONLY JSON."),
                HumanMessage(content=prompt_str)
            ])
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            scores = json.loads(raw)
            score_dict = {item.get("id"): float(item.get("relevance", 0)) for item in scores if item.get("id")}
        except Exception:
            # Fallback to RRF scores if LLM formatting errors
            score_dict = {c["rule_id"]: c["score"] for c in candidates}

        # Sort candidates by re-rank score
        reranked = []
        for c in candidates:
            c_copy = dict(c)
            c_copy["rerank_score"] = score_dict.get(c["rule_id"], 0.0)
            reranked.append(c_copy)

        reranked.sort(key=lambda x: (x.get("rerank_score", 0), x["score"]), reverse=True)
        return reranked[:top_k]

    # ── Method 5: Format-Aware Filtered Hybrid + Re-Ranking ──────────────────
    def retrieve_format_filtered_hybrid(self, query: str, sample_meta: Dict[str, Any], top_k: int = 6) -> List[Dict[str, Any]]:
        """
        Production-grade method:
        Applies domain constraints before retrieval:
          - If long video (>60s) or in-stream format, prune bumper ad limit rule (YT-SPEC-003)
          - If standard video, prune non-skippable 15s limit (YT-SPEC-002)
          - If vertical Shorts, prioritize Shorts safe zones (YT-SPEC-004)
        """
        ad_format = sample_meta.get("ad_format", "in_stream")
        duration = sample_meta.get("duration_seconds", 0.0)

        # Run hybrid retrieval on all candidates
        candidates = self.retrieve_hybrid_rrf(query, top_k=10)

        filtered = []
        for c in candidates:
            rid = c["rule_id"]
            # Prune bumper ad limit if video is clearly not a bumper ad
            if rid == "YT-SPEC-003" and (ad_format in ["in_stream", "organic_video"] or duration > 15.0):
                continue
            # Prune non-skippable limit if video is skippable in-stream
            if rid == "YT-SPEC-002" and ad_format == "in_stream":
                continue
            # Prune Shorts vertical safe zone if video is horizontal 16:9
            if rid == "YT-SPEC-004" and sample_meta.get("aspect_ratio") == "16:9":
                continue
            filtered.append(c)

        # Prioritize shorts rule if shorts format
        if ad_format == "shorts":
            for c in filtered:
                if c["rule_id"] == "YT-SPEC-004":
                    c["score"] += 1.0

        filtered.sort(key=lambda x: x["score"], reverse=True)
        return filtered[:top_k]

    # ── Method 6: Flat / Exact Brute-Force k-NN (No Graph Approximation) ─────
    def retrieve_flat_exact(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """
        Exact brute-force cosine similarity over all raw vectors (O(N)).
        Serves as the theoretical ground-truth upper bound for vector similarity.
        """
        q_emb = self.ef([query])[0]
        scored = []
        for r in self.all_rules:
            # get document embedding via collection
            doc_res = self.collection.get(ids=[r["rule_id"]], include=["embeddings"])
            if doc_res is not None and "embeddings" in doc_res and doc_res["embeddings"] is not None and len(doc_res["embeddings"]) > 0:
                d_emb = doc_res["embeddings"][0]
                # compute exact cosine similarity
                dot = sum(a * b for a, b in zip(q_emb, d_emb))
                norm_q = math.sqrt(sum(a * a for a in q_emb))
                norm_d = math.sqrt(sum(b * b for b in d_emb))
                cos = dot / max(1e-9, (norm_q * norm_d))
                scored.append({
                    "rule_id": r["rule_id"],
                    "title": r.get("title", ""),
                    "category": r.get("category", ""),
                    "severity": r.get("severity", ""),
                    "source": r.get("source", ""),
                    "score": round(cos, 4),
                    "text": r["text"]
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ── Method 7: True Cross-Encoder Re-Ranking (MiniLM Cross-Attention) ─────
    def retrieve_cross_encoder_rerank(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """
        Takes candidate rules from Hybrid RRF and applies joint cross-attention
        using cross-encoder/ms-marco-MiniLM-L-6-v2 in milliseconds.
        """
        candidates = self.retrieve_hybrid_rrf(query, top_k=10)
        tok, mod = self.cross_encoder

        pairs = [[query[:400], c["text"][:500]] for c in candidates]
        inputs = tok(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)

        with torch.no_grad():
            scores = mod(**inputs).logits.squeeze(-1).tolist()

        if isinstance(scores, float):
            scores = [scores]

        reranked = []
        for c, s in zip(candidates, scores):
            c_copy = dict(c)
            c_copy["cross_encoder_score"] = round(s, 4)
            c_copy["score"] = round(s, 4)
            reranked.append(c_copy)

        reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return reranked[:top_k]

    # ── Method 8: Production Format-Aware Filtered + Cross-Encoder ───────────
    def retrieve_filtered_cross_encoder(self, query: str, sample_meta: Dict[str, Any], top_k: int = 6) -> List[Dict[str, Any]]:
        """
        Complete production pipeline:
        Format metadata filter + Hybrid candidate generation + Cross-Encoder re-ranking.
        """
        candidates = self.retrieve_format_filtered_hybrid(query, sample_meta, top_k=8)
        tok, mod = self.cross_encoder

        pairs = [[query[:400], c["text"][:500]] for c in candidates]
        inputs = tok(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)

        with torch.no_grad():
            scores = mod(**inputs).logits.squeeze(-1).tolist()

        if isinstance(scores, float):
            scores = [scores]

        reranked = []
        for c, s in zip(candidates, scores):
            c_copy = dict(c)
            c_copy["cross_encoder_score"] = round(s, 4)
            c_copy["score"] = round(s, 4)
            reranked.append(c_copy)

        reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return reranked[:top_k]
