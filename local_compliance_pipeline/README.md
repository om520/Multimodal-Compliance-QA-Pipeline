# Brand Guardian — Multimodal Compliance QA Pipeline

> **Local & Cloud Video Compliance Auditing with Exhaustive RAG Retrieval Evaluation, HNSW vs. Flat Exact k-NN Benchmarking, and LangGraph Orchestration.**

---

## 1. Quick Navigation
* **RAG Retrieval Suite & Evaluation:** [`rag_retrieval_and_database/`](rag_retrieval_and_database/README.md)
* **Preserved Azure Cloud Pipeline:** [`azure_compliance_pipeline/`](../azure_compliance_pipeline/README.md)
* **Retrieval Benchmark Notebook:** [`compliance_rag_benchmark_and_evaluation.ipynb`](compliance_rag_benchmark_and_evaluation.ipynb)
* **4-Video Audit Traceability Notebook:** [`compliance_audit_4_videos_traceability.ipynb`](compliance_audit_4_videos_traceability.ipynb)
* **Chunking & Vector Space Analysis:** [`compliance_rules_chunking_and_retrieval.ipynb`](compliance_rules_chunking_and_retrieval.ipynb)

---

## 2. What It Does

1. **Extracts Multimodal Signals:** Transcribes spoken audio (Whisper), extracts on-screen text & disclaimers (PaddleOCR), and generates visual scene tags (BLIP).
2. **Retrieves Statutory Rules:** Ingests regulatory PDFs (FTC Endorsement Guides, YouTube Ad Specs) and uses a 2-stage retrieval pipeline (Format Filtering + Hybrid RRF + Cross-Encoder).
3. **Audits & Generates Traceability Reports:** Evaluates compliance violations via Ollama (`qwen2.5:7b-instruct`) and outputs structured reports with timestamp offsets and rule clause citations.

---

## 3. Key Investigation: HNSW vs. Flat Exact k-NN on Small Corpora

### The Small-Dataset Reality
* For our pilot statutory corpus of regulatory clauses, **Flat (Exact Brute-Force) Cosine k-NN is mathematically optimal**. It computes exact distances in `<1ms`, has zero graph construction overhead, and guarantees 100% exact recall.
* **HNSW** is an approximate graph algorithm designed for hundreds of thousands of vectors. Using it on small datasets is technically unnecessary and could theoretically risk graph-pruning errors ($M$ edge limits).

### Why ChromaDB Defaults to HNSW & Why We Tested Both
ChromaDB configures HNSW by default to enable scaling out-of-the-box. To verify that ChromaDB's default HNSW was not causing any approximation loss or missing small legal clauses, we benchmarked a **Flat Exact k-NN oracle** against **HNSW**:

| Metric | Flat (Exact k-NN Oracle) | HNSW (Default Vector DB) | Trade-Off / Degradation |
| :--- | :---: | :---: | :---: |
| **Recall@5** | **0.7400** | **0.7400** | **0.00% (Identical)** |
| **Precision@5** | **0.2400** | **0.2400** | **0.00% (Identical)** |
| **MRR** | **0.5450** | **0.5450** | **0.00% (Identical)** |
| **Latency** | ~22 ms | ~18 ms | HNSW ~4ms faster |

**Conclusion:** Using ChromaDB's default HNSW introduces **zero recall penalty** on our dataset while ensuring the pipeline is architecture-ready to scale to thousands of multi-jurisdictional compliance clauses.

---

## 4. Comprehensive Retrieval & Re-ranking Evaluation Matrix

We evaluated **7 distinct retrieval combinations** against our multi-label statutory Golden Dataset ($N=10$):

| # | Architecture Configuration | Retrieval / Indexing | Re-ranking / Filtering | Recall@5 | Precision@5 | MRR | Latency |
|---|---|---|---|:---:|:---:|:---:|:---:|
| **1** | Dense Vector | HNSW (`nomic-embed-text`) | None (Cosine Sim) | 0.7400 | 0.2400 | 0.5450 | ~18 ms |
| **2** | Flat Exact | Brute-force Cosine | None (Exact k-NN) | 0.7400 | 0.2400 | 0.5450 | ~22 ms |
| **3** | Sparse Lexical | Inverted Index (Okapi BM25) | None (BM25 Score) | 0.7733 | 0.2400 | **0.7150** | **< 1 ms** |
| **4** | Hybrid RRF | HNSW + BM25 | Reciprocal Rank Fusion ($k=60$) | 0.7733 | 0.2600 | 0.5950 | ~19 ms |
| **5** | Neural Re-ranked | HNSW Bi-Encoder | Cross-Encoder (`ms-marco-MiniLM`) | 0.7400 | 0.2400 | 0.5450 | ~28 ms |
| **6** | Format Filtered | HNSW + BM25 Hybrid | Statutory Format Rule Filter | 0.7933 | 0.2800 | 0.7083 | ~20 ms |
| **7** | **Production Pipeline** | **Format-Filtered Hybrid** | **Cross-Encoder Re-ranker** | **0.8933** | **0.3000** | 0.4233 | ~35 ms |

### Generation Quality (TruLens RAG Triad)
* **Groundedness:** **0.95 / 1.00** *(All violation claims cite extracted OCR/transcript)*
* **Faithfulness:** **0.90 / 1.00** *(No hallucinated regulatory mandates)*
* **Context Relevance:** **0.86 / 1.00** *(Filter eliminates non-applicable modalities)*

---

## 5. Running the Pipeline

### One-Time Setup (Windows PowerShell)
```powershell
uv sync
.\setup_local_services.bat
```

### Reproducing the RAG Benchmark
```powershell
uv run python rag_retrieval_and_database/run_benchmark.py
```

### Starting the Web UI & API
```powershell
.\start_server.bat
```
* **Browser UI:** <http://127.0.0.1:8000>
* **Interactive OpenAPI Docs:** <http://127.0.0.1:8000/docs>

---

## 6. Directory Layout

```text
local_compliance_pipeline/
├── rag_retrieval_and_database/      ← Self-contained RAG benchmark suite & datasets
├── backend/
│   ├── src/api/server.py            ← FastAPI routes & UI entry point
│   ├── src/graph/workflow.py        ← LangGraph orchestration
│   └── src/services/video_indexer.py← Local Whisper, PaddleOCR, and BLIP
├── frontend/                        ← Interactive browser interface
├── chroma_db/                       ← Persistent vector database
├── audit_runs/                      ← JSON evidence records of completed audits
├── compliance_rag_benchmark_and_evaluation.ipynb
├── compliance_audit_4_videos_traceability.ipynb
└── compliance_rules_chunking_and_retrieval.ipynb
```
