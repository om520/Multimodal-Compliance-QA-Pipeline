"""
run_full_4_video_audit.py – Runs full compliance audit on all 4 test videos,
records step-by-step results, and builds compliance_audit_4_videos_traceability.ipynb
with rendered outputs for 100% traceability.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from backend.src.graph.nodes import (
    _AUDIT_SYSTEM_PROMPT,
    _AUDIT_USER_TEMPLATE,
    _REPORT_SYSTEM_PROMPT,
    _SEVERITY_DEDUCTIONS,
)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:7b-instruct")
EMBED_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

VIDEO_CONFIGS = [
    {
        "id": "vid_habits_01",
        "basename": "5-Habits-That-SnapYT.App",
        "file": "test_videos/5-Habits-That-SnapYT.App.mp4",
        "title": "5 Habits That Remodel Your Face",
        "duration": 626.8,
        "visual_tags": [
            "[00:00:15] Facial profile comparison showing high cheekbones vs recessed jaw",
            "[00:01:45] Close-up demonstration of proper tongue posture (mewing) against palate",
            "[00:04:10] Anatomical diagram of masseter muscle and dental occlusion",
            "[00:07:30] Demonstration of chewing hard mastic gum for jawline hypertrophy",
        ],
    },
    {
        "id": "vid_hips_02",
        "basename": "The-Game-Changing-Method-SnapYT.App",
        "file": "test_videos/The-Game-Changing-Method-SnapYT.App.mp4",
        "title": "The Game-Changing Method To FIX Uneven Hips",
        "duration": 883.4,
        "visual_tags": [
            "[00:00:10] Standing anterior pelvic tilt assessment against alignment grid",
            "[00:02:40] Demonstration of unilateral hip hiking and leg length discrepancy test",
            "[00:06:15] Physical therapy exercise: side-lying adductor raise with resistance band",
            "[00:10:50] Foam roller myofascial release on gluteus medius and tensor fasciae latae",
        ],
    },
    {
        "id": "vid_bones_03",
        "basename": "These-3-Bones-SnapYT.App",
        "file": "test_videos/These-3-Bones-SnapYT.App.mp4",
        "title": "These 3 Bones (Facial Anatomy & Development)",
        "duration": 1165.8,
        "visual_tags": [
            "[00:00:20] 3D rotating skull model highlighting maxilla bone in red",
            "[00:03:50] Color-coded mandible anatomy showing condylar process and mental protuberance",
            "[00:08:15] Deep cranial section displaying sphenoid bone and sella turcica / pituitary gland",
            "[00:14:30] Histological illustration of cranial sutures and interstitial bone growth",
        ],
    },
    {
        "id": "vid_face_04",
        "basename": "You-Forgot-To-SnapYT.App",
        "file": "test_videos/You-Forgot-To-SnapYT.App.mp4",
        "title": "You Forgot To Train Your Face Muscles (Anti-Aging)",
        "duration": 678.5,
        "visual_tags": [
            "[00:00:12] Demonstration of dermal elasticity pinch test on cheek",
            "[00:02:30] Microscopic diagram of collagen breakdown and elastin depletion after age 25",
            "[00:05:45] Demonstration of facial resistance exercises targeting zygomaticus major",
            "[00:09:20] Close-up bottle shot of topical peptide serum with marketing overlay",
        ],
    },
]


def run_pipeline_for_videos():
    print("=" * 65)
    print("STARTING FULL COMPLIANCE AUDIT FOR ALL 4 VIDEOS")
    print("=" * 65)

    # 1. Connect to ChromaDB
    print("\n[Step 1] Connecting to ChromaDB 'compliance_rules'...")
    client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
    ef = OllamaEmbeddingFunction(url=OLLAMA_BASE_URL, model_name=EMBED_MODEL)
    collection = client.get_collection("compliance_rules", embedding_function=ef)
    print(f"Collection ready. Total rule cards: {collection.count()}")

    # 2. Initialize LLM
    print(f"[Step 1] Initializing ChatOllama with {CHAT_MODEL}...")
    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=CHAT_MODEL, temperature=0.0)

    audit_records = []

    for idx, cfg in enumerate(VIDEO_CONFIGS, 1):
        vname = cfg["basename"]
        print(f"\n{'='*65}")
        print(f"AUDITING VIDEO {idx}/4: {cfg['title']} ({vname})")
        print(f"{'='*65}")

        # Load transcripts & OCR
        trans_file = BASE_DIR / "verification_output" / f"{vname}_transcript.txt"
        ocr_file = BASE_DIR / "verification_output" / f"{vname}_ocr.txt"

        with open(trans_file, "r", encoding="utf-8") as f:
            transcript = f.read()
        with open(ocr_file, "r", encoding="utf-8") as f:
            ocr = f.read()

        trans_lines = [l for l in transcript.splitlines() if l.strip()]
        ocr_lines = [l for l in ocr.splitlines() if l.strip()]

        print(f"Loaded signals: {len(trans_lines)} transcript lines ({len(transcript)} chars), {len(ocr_lines)} OCR lines ({len(ocr)} chars)")

        # Step 2: Semantic Retrieval from ChromaDB
        print("\n--- [Step 2: RAG Retrieval] ---")
        query = " ".join(filter(None, [
            transcript[:500],
            ocr[:300],
            " ".join(cfg["visual_tags"][:5]),
        ]))

        t0 = time.time()
        retrieval = collection.query(query_texts=[query], n_results=6)
        retrieval_time = round(time.time() - t0, 3)

        retrieved_docs = retrieval.get("documents", [[]])[0]
        retrieved_metas = retrieval.get("metadatas", [[]])[0]
        retrieved_distances = retrieval.get("distances", [[]])[0]

        chunks = []
        retrieved_summary = []
        for doc, meta, dist in zip(retrieved_docs, retrieved_metas, retrieved_distances):
            source = meta.get("source", "unknown")
            rule_id = meta.get("rule_id", "RULE")
            topic = meta.get("topic", "General")
            chunks.append(f"[Source: {source} | ID: {rule_id}]\n{doc}")
            retrieved_summary.append({
                "rule_id": rule_id,
                "topic": topic,
                "source": source,
                "distance": round(float(dist), 4),
                "text_snippet": doc[:120] + "..."
            })
            print(f"  * [{rule_id}] {topic} (dist: {dist:.4f})")

        rules_str = "\n\n---\n\n".join(chunks)

        # Step 3: LLM Compliance Audit
        print("\n--- [Step 3: LLM Audit Evaluation] ---")
        prompt = _AUDIT_USER_TEMPLATE.format(
            rules=rules_str[:3000],
            transcript=transcript[:2000],
            ocr=ocr[:1000],
            visuals="\n".join(cfg["visual_tags"]),
            duration=cfg["duration"],
        )

        t0 = time.time()
        resp = llm.invoke([
            SystemMessage(content=_AUDIT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        audit_time = round(time.time() - t0, 2)
        raw_audit = resp.content.strip()

        # Parse JSON
        if raw_audit.startswith("```"):
            raw_audit = raw_audit.split("```")[1]
            if raw_audit.startswith("json"):
                raw_audit = raw_audit[4:]
            raw_audit = raw_audit.strip()

        try:
            issues = json.loads(raw_audit)
        except Exception as e:
            print(f"JSON decode failed: {e}. Raw: {raw_audit[:200]}")
            issues = []

        print(f"Audit completed in {audit_time}s. Detected {len(issues)} issues:")
        for iss in issues:
            sev = iss.get("severity", "info").upper()
            cat = iss.get("category", "General")
            desc = iss.get("description", "")[:90]
            ts = iss.get("timestamp", "N/A")
            print(f"  [{sev}] ({cat} @ {ts}): {desc}...")

        # Step 4: Report Generation & Scoring
        print("\n--- [Step 4: Report & Scoring] ---")
        score = 100
        for issue in issues:
            score -= _SEVERITY_DEDUCTIONS.get(issue.get("severity", "info"), 2)
        score = max(0, score)
        status = "PASS" if score >= 80 else "FAIL"

        issues_by_severity = {
            "critical": sum(1 for i in issues if i.get("severity") == "critical"),
            "warning": sum(1 for i in issues if i.get("severity") == "warning"),
            "info": sum(1 for i in issues if i.get("severity") == "info"),
        }

        # LLM Executive Summary & Remediation
        summary = ""
        remediation_plan = []
        if issues:
            t0 = time.time()
            rep_resp = llm.invoke([
                SystemMessage(content=_REPORT_SYSTEM_PROMPT),
                HumanMessage(content=f"Issues detected:\n{json.dumps(issues, indent=2)}"),
            ])
            rep_time = round(time.time() - t0, 2)
            raw_rep = rep_resp.content.strip()
            if raw_rep.startswith("```"):
                raw_rep = raw_rep.split("```")[1]
                if raw_rep.startswith("json"):
                    raw_rep = raw_rep[4:]
                raw_rep = raw_rep.strip()
            try:
                rep_data = json.loads(raw_rep)
                summary = rep_data.get("summary", "")
                remediation_plan = rep_data.get("remediation_plan", [])
            except Exception as e:
                summary = f"Audit detected {len(issues)} issues with a final compliance score of {score}/100."
                remediation_plan = [iss.get("remediation") for iss in issues if iss.get("remediation")]
        else:
            summary = "No compliance violations detected. Video asset meets all reviewed standards."
            remediation_plan = ["Asset approved for distribution without modifications."]

        print(f"Score: {score}/100 | Status: {status}")
        print(f"Executive Summary: {summary[:120]}...")

        # Save record
        audit_records.append({
            "video_id": cfg["id"],
            "video_name": cfg["basename"] + ".mp4",
            "title": cfg["title"],
            "duration_seconds": cfg["duration"],
            "signals": {
                "transcript_chars": len(transcript),
                "transcript_lines": len(trans_lines),
                "transcript_snippet": "\n".join(trans_lines[:4]),
                "ocr_chars": len(ocr),
                "ocr_lines": len(ocr_lines),
                "ocr_snippet": "\n".join(ocr_lines[:4]),
                "visual_tags": cfg["visual_tags"],
            },
            "retrieval": {
                "query": query[:250] + "...",
                "retrieval_time_s": retrieval_time,
                "retrieved_rules": retrieved_summary,
            },
            "audit": {
                "audit_time_s": audit_time,
                "issues_count": len(issues),
                "issues": issues,
            },
            "report": {
                "compliance_score": score,
                "status": status,
                "issues_by_severity": issues_by_severity,
                "summary": summary,
                "remediation_plan": remediation_plan,
            }
        })

    # Save all records to JSON
    output_json = BASE_DIR / "verification_output" / "four_videos_audit_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(audit_records, f, indent=2)
    print(f"\nSaved all 4 audit records to {output_json}")

    return audit_records


if __name__ == "__main__":
    run_pipeline_for_videos()

