# Compliance RAG Retrieval & Database Suite

A self-contained module containing all database indexing, search paradigms, re-ranking architectures, and benchmark evaluation tools for the Compliance QA Pipeline.

---

## 1. Directory Structure

```
rag_retrieval_and_database/
├── data/                                 # Regulatory source documents
│   ├── 1001a-influencer-guide-508_1.pdf  # FTC Influencer Endorsement Guides
│   └── YouTube_Ad_Specs_Guide_2026.pdf   # YouTube Ad Technical Specs Guide
├── lean_dynamic_chunker.py               # Indexes 12 lean semantic rule cards into ChromaDB
├── retrieval_and_reranking.py            # Complete 7-method retrieval & re-ranking suite
├── evaluator.py                          # Metric computation (P@5, R@5, MRR, Groundedness, Faithfulness)
├── run_benchmark.py                      # Automated benchmark CLI
├── golden_dataset.json                   # 10 multimodal test samples with injected mistakes
├── golden_labels.json                    # Ground-truth labels & remediation requirements
└── benchmark_results.json                # Persisted empirical benchmark outputs
```

---

## 2. Implemented Retrieval Paradigms

1. **Dense Vector Search (HNSW):** Approximate Nearest Neighbor search on 768-dim `nomic-embed-text` vectors in ChromaDB (`cosine` metric).
2. **Flat / Exact k-NN:** Brute-force exhaustive cosine scan across 100% of raw un-indexed embeddings (theoretical ground-truth upper bound).
3. **Sparse Lexical (Okapi BM25):** Pure-Python inverted index matching exact statutory keywords (`#ad`, `#sp`, `safe zone`, `6.0s`).
4. **Hybrid Search (Reciprocal Rank Fusion):** Fuses Dense + BM25 ranks via $\text{RRF}(d) = \sum \frac{1}{60 + \text{rank}(d)}$.
5. **Bi-Encoder + Cross-Encoder Re-Ranking:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for token-level joint cross-attention scoring in ~28ms.
6. **Format-Aware Filtered Hybrid:** Dynamically prunes format-incompatible rules (e.g. excludes 6s Bumper limits on long-form video).
7. **Format-Filtered + Cross-Encoder (Production):** Blends domain priors, hybrid candidates, and cross-encoder re-ranking for **89.3% Recall@5** and **100% Faithfulness**.

---

## 3. Quickstart Commands

```bash
# 1. Re-index compliance PDFs into ChromaDB
python lean_dynamic_chunker.py

# 2. Run the 7-method comparative benchmark
python run_benchmark.py
```

