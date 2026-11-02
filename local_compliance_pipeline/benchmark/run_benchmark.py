"""
run_benchmark.py – End-to-end benchmark execution comparing 5 retrieval & auditing methods
against golden_dataset.json and golden_labels.json.

Generates:
  - Precision@5, Recall@5, MRR across retrieval methods
  - Audit Precision, Recall, F1-Score, Groundedness, Faithfulness
  - Structured output benchmark_results.json
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from benchmark.retrieval_and_reranking import ComplianceRetrievalSuite
from benchmark.evaluator import evaluate_retrieval, evaluate_audit_issues
from backend.src.graph.nodes import _AUDIT_SYSTEM_PROMPT, _AUDIT_USER_TEMPLATE
from langchain_core.messages import SystemMessage, HumanMessage


def run_benchmark():
    benchmark_dir = ROOT_DIR / "benchmark"
    dataset_file = benchmark_dir / "golden_dataset.json"
    labels_file = benchmark_dir / "golden_labels.json"

    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)["samples"]

    with open(labels_file, "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]

    print("=" * 80)
    print("      MULTIMODAL COMPLIANCE RAG RETRIEVAL & AUDIT BENCHMARK")
    print(f"      Total Golden Samples: {len(dataset)} | Ground Truth Labels: {len(labels)}")
    print("=" * 80)

    engine = ComplianceRetrievalSuite()
    methods = [
        ("Dense HNSW (nomic-embed-text)", "dense"),
        ("Flat Exact k-NN (Brute-Force Cosine)", "flat_exact"),
        ("Sparse Lexical (Okapi BM25)", "sparse"),
        ("Hybrid RRF (Dense HNSW + BM25)", "hybrid_rrf"),
        ("Bi-Encoder + Cross-Encoder (MiniLM)", "cross_encoder"),
        ("Format-Aware Filtered Hybrid", "filtered_hybrid"),
        ("Format-Filtered + Cross-Encoder (Production)", "filtered_cross_encoder"),
    ]

    method_metrics = {m_key: {"p_at_5": [], "r_at_5": [], "mrr": []} for _, m_key in methods}
    sample_retrieval_traces = []

    print("\n--- PHASE 1: RETRIEVAL METHODOLOGY BENCHMARK ---")

    for sample in dataset:
        sid = sample["sample_id"]
        meta = labels[sid]
        expected_ids = meta["expected_rule_ids"]

        query = " ".join(filter(None, [
            sample.get("transcript", "")[:500],
            sample.get("ocr_text", "")[:300],
            " ".join(sample.get("visual_tags", [])[:5])
        ]))

        sample_trace = {
            "sample_id": sid,
            "title": sample["title"],
            "expected_rules": expected_ids,
            "method_results": {}
        }

        for m_name, m_key in methods:
            t0 = time.time()
            if m_key == "dense":
                retrieved = engine.retrieve_dense(query, top_k=5)
            elif m_key == "flat_exact":
                retrieved = engine.retrieve_flat_exact(query, top_k=5)
            elif m_key == "sparse":
                retrieved = engine.retrieve_sparse(query, top_k=5)
            elif m_key == "hybrid_rrf":
                retrieved = engine.retrieve_hybrid_rrf(query, top_k=5)
            elif m_key == "cross_encoder":
                retrieved = engine.retrieve_cross_encoder_rerank(query, top_k=5)
            elif m_key == "filtered_hybrid":
                retrieved = engine.retrieve_format_filtered_hybrid(query, sample, top_k=5)
            elif m_key == "filtered_cross_encoder":
                retrieved = engine.retrieve_filtered_cross_encoder(query, sample, top_k=5)

            lat = round((time.time() - t0) * 1000, 1)
            r_ids = [r["rule_id"] for r in retrieved]
            eval_res = evaluate_retrieval(r_ids, expected_ids, k=5)

            method_metrics[m_key]["p_at_5"].append(eval_res["precision_at_k"])
            method_metrics[m_key]["r_at_5"].append(eval_res["recall_at_k"])
            method_metrics[m_key]["mrr"].append(eval_res["mrr"])

            sample_trace["method_results"][m_key] = {
                "retrieved_ids": r_ids,
                "latency_ms": lat,
                "metrics": eval_res
            }

        sample_retrieval_traces.append(sample_trace)
        print(f"  Processed {sid}: {sample['title'][:40]:<40} -> Dense P@5: {sample_trace['method_results']['dense']['metrics']['precision_at_k']:.2f} | Filtered P@5: {sample_trace['method_results']['filtered_hybrid']['metrics']['precision_at_k']:.2f}")

    # Summary table for Retrieval
    print("\n" + "=" * 80)
    print("PHASE 1 RETRIEVAL BENCHMARK RESULTS (Averaged across 10 Golden Samples)")
    print("=" * 80)
    print(f"{'Retrieval Method':<40} | {'Precision@5':<12} | {'Recall@5':<10} | {'MRR':<8}")
    print("-" * 80)

    retrieval_summary = {}
    for m_name, m_key in methods:
        avg_p = sum(method_metrics[m_key]["p_at_5"]) / len(dataset)
        avg_r = sum(method_metrics[m_key]["r_at_5"]) / len(dataset)
        avg_mrr = sum(method_metrics[m_key]["mrr"]) / len(dataset)
        retrieval_summary[m_key] = {
            "name": m_name,
            "mean_precision_at_5": round(avg_p, 4),
            "mean_recall_at_5": round(avg_r, 4),
            "mean_mrr": round(avg_mrr, 4)
        }
        print(f"{m_name:<40} | {avg_p:<12.4f} | {avg_r:<10.4f} | {avg_mrr:<8.4f}")

    # ── PHASE 2: AUDIT & FAITHFULNESS EVALUATION ─────────────────────────────
    print("\n--- PHASE 2: AUDIT EVALUATION (Baseline Dense vs Format-Filtered Hybrid) ---")
    print("Testing LLM Audit on representative positive violations & negative controls...")

    test_samples = [s for s in dataset if s["sample_id"] in ["GOLDEN_01", "GOLDEN_04", "GOLDEN_05", "GOLDEN_08", "GOLDEN_10"]]
    audit_eval_results = []

    # Check for cached audit traces to enable instantaneous Phase 2 evaluation
    cached_audits = {}
    results_file = benchmark_dir / "benchmark_results.json"
    if results_file.exists():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                old = json.load(f)
                cached_audits = {t["sample_id"]: t for t in old.get("sample_audit_eval_traces", [])}
        except Exception:
            cached_audits = {}

    for sample in test_samples:
        sid = sample["sample_id"]
        lbl = labels[sid]
        query = sample.get("transcript", "")[:500] + " " + sample.get("ocr_text", "")[:300]

        dense_rules = engine.retrieve_dense(query, top_k=5)
        filtered_rules = engine.retrieve_format_filtered_hybrid(query, sample, top_k=5)
        cross_rules = engine.retrieve_filtered_cross_encoder(query, sample, top_k=5)

        if sid in cached_audits:
            issues_dense = cached_audits[sid]["dense_audit"]["issues"]
            issues_filtered = cached_audits[sid]["filtered_audit"]["issues"]
        else:
            # Run Baseline Dense Retrieval
            dense_str = "\n\n---\n\n".join([f"[Source: {r.get('source')}]\n{r['text']}" for r in dense_rules])
            prompt_dense = _AUDIT_USER_TEMPLATE.format(
                rules=dense_str[:3000],
                transcript=sample.get("transcript", "")[:2000],
                ocr=sample.get("ocr_text", "")[:1000],
                visuals="\n".join(sample.get("visual_tags", [])),
                duration=sample.get("duration_seconds", 0.0)
            )
            resp_dense = engine.llm.invoke([
                SystemMessage(content=_AUDIT_SYSTEM_PROMPT),
                HumanMessage(content=prompt_dense)
            ])
            raw_dense = resp_dense.content.strip()
            if raw_dense.startswith("```"):
                raw_dense = raw_dense.split("```")[1]
                if raw_dense.startswith("json"):
                    raw_dense = raw_dense[4:]
                raw_dense = raw_dense.strip()
            try:
                issues_dense = json.loads(raw_dense)
            except Exception:
                issues_dense = []

            # Run Format-Filtered Hybrid Retrieval
            filtered_str = "\n\n---\n\n".join([f"[Source: {r.get('source')}]\n{r['text']}" for r in filtered_rules])
            prompt_filtered = _AUDIT_USER_TEMPLATE.format(
                rules=filtered_str[:3000],
                transcript=sample.get("transcript", "")[:2000],
                ocr=sample.get("ocr_text", "")[:1000],
                visuals="\n".join(sample.get("visual_tags", [])),
                duration=sample.get("duration_seconds", 0.0)
            )
            resp_filtered = engine.llm.invoke([
                SystemMessage(content=_AUDIT_SYSTEM_PROMPT),
                HumanMessage(content=prompt_filtered)
            ])
            raw_filtered = resp_filtered.content.strip()
            if raw_filtered.startswith("```"):
                raw_filtered = raw_filtered.split("```")[1]
                if raw_filtered.startswith("json"):
                    raw_filtered = raw_filtered[4:]
                raw_filtered = raw_filtered.strip()
            try:
                issues_filtered = json.loads(raw_filtered)
            except Exception:
                issues_filtered = []

        eval_dense = evaluate_audit_issues(
            issues_dense, lbl, sample, [r["rule_id"] for r in dense_rules]
        )
        eval_filtered = evaluate_audit_issues(
            issues_filtered, lbl, sample, [r["rule_id"] for r in filtered_rules]
        )

        audit_eval_results.append({
            "sample_id": sid,
            "title": sample["title"],
            "is_violation": lbl["is_violation"],
            "dense_audit": {
                "issues": issues_dense,
                "metrics": eval_dense
            },
            "filtered_audit": {
                "issues": issues_filtered,
                "metrics": eval_filtered
            }
        })

        print(f"\n  [{sid}] {sample['title']}")
        print(f"    Baseline Dense:       Prec={eval_dense['precision']:.2f} | Rec={eval_dense['recall']:.2f} | Grounded={eval_dense['groundedness']:.2f} | Faithful={eval_dense['faithfulness']:.2f}")
        print(f"    Format-Filtered Hyb:  Prec={eval_filtered['precision']:.2f} | Rec={eval_filtered['recall']:.2f} | Grounded={eval_filtered['groundedness']:.2f} | Faithful={eval_filtered['faithfulness']:.2f}")

    # Summary of Audit Evaluation
    dense_p = sum(r["dense_audit"]["metrics"]["precision"] for r in audit_eval_results) / len(audit_eval_results)
    dense_r = sum(r["dense_audit"]["metrics"]["recall"] for r in audit_eval_results) / len(audit_eval_results)
    dense_f1 = sum(r["dense_audit"]["metrics"]["f1_score"] for r in audit_eval_results) / len(audit_eval_results)
    dense_ground = sum(r["dense_audit"]["metrics"]["groundedness"] for r in audit_eval_results) / len(audit_eval_results)
    dense_faith = sum(r["dense_audit"]["metrics"]["faithfulness"] for r in audit_eval_results) / len(audit_eval_results)

    filt_p = sum(r["filtered_audit"]["metrics"]["precision"] for r in audit_eval_results) / len(audit_eval_results)
    filt_r = sum(r["filtered_audit"]["metrics"]["recall"] for r in audit_eval_results) / len(audit_eval_results)
    filt_f1 = sum(r["filtered_audit"]["metrics"]["f1_score"] for r in audit_eval_results) / len(audit_eval_results)
    filt_ground = sum(r["filtered_audit"]["metrics"]["groundedness"] for r in audit_eval_results) / len(audit_eval_results)
    filt_faith = sum(r["filtered_audit"]["metrics"]["faithfulness"] for r in audit_eval_results) / len(audit_eval_results)

    print("\n" + "=" * 80)
    print("PHASE 2 AUDIT & GENERATION QUALITY COMPARISON")
    print("=" * 80)
    print(f"{'Pipeline Configuration':<35} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8} | {'Groundedness':<12} | {'Faithfulness'}")
    print("-" * 95)
    print(f"{'Baseline Dense Retrieval + LLM':<35} | {dense_p:<10.4f} | {dense_r:<8.4f} | {dense_f1:<8.4f} | {dense_ground:<12.4f} | {dense_faith:.4f}")
    print(f"{'Format-Filtered Hybrid + LLM':<35} | {filt_p:<10.4f} | {filt_r:<8.4f} | {filt_f1:<8.4f} | {filt_ground:<12.4f} | {filt_faith:.4f}")

    # Save comprehensive results
    output_benchmark = {
        "benchmark_summary": {
            "retrieval_methods": retrieval_summary,
            "audit_comparison": {
                "baseline_dense": {
                    "precision": round(dense_p, 4),
                    "recall": round(dense_r, 4),
                    "f1_score": round(dense_f1, 4),
                    "groundedness": round(dense_ground, 4),
                    "faithfulness": round(dense_faith, 4)
                },
                "format_filtered_hybrid": {
                    "precision": round(filt_p, 4),
                    "recall": round(filt_r, 4),
                    "f1_score": round(filt_f1, 4),
                    "groundedness": round(filt_ground, 4),
                    "faithfulness": round(filt_faith, 4)
                }
            }
        },
        "sample_retrieval_traces": sample_retrieval_traces,
        "sample_audit_eval_traces": audit_eval_results
    }

    results_file = benchmark_dir / "benchmark_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output_benchmark, f, indent=2)

    print(f"\n[OK] Benchmark results saved to: {results_file}")
    return output_benchmark


if __name__ == "__main__":
    run_benchmark()

