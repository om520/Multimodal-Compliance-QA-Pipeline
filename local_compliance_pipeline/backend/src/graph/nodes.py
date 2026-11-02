"""
nodes.py – LangGraph node implementations for the local Compliance QA Pipeline.

Replaces all Azure services with local equivalents:
  - Azure Video Indexer  →  LocalMediaService (faster-whisper + PaddleOCR + BLIP)
  - Azure OpenAI         →  Ollama via langchain-ollama
  - Azure AI Search      →  ChromaDB (local vector store)

Node execution order (defined in workflow.py):
  index_video → retrieve_compliance_rules → audit_compliance → generate_report
"""

import json
import logging
import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from backend.src.graph.state import ComplianceIssue, VideoAudioState
from backend.src.services.video_indexer import LocalMediaService

load_dotenv()

logger = logging.getLogger("brand-guardian")

# ─────────────────────────────────────────────────────────────────────────────
# Shared local model clients
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm() -> ChatOllama:
    return ChatOllama(
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        model=os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:7b-instruct"),
        temperature=0,
    )


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        model=os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    )


def _get_vectorstore():
    """Return the ChromaDB collection used as the local vector store."""
    import chromadb
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "compliance_rules")

    client = chromadb.PersistentClient(path=persist_dir)
    ef = OllamaEmbeddingFunction(
        url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        model_name=os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    )
    return client.get_or_create_collection(collection_name, embedding_function=ef)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 – Video Ingestion & Local Analysis
# ─────────────────────────────────────────────────────────────────────────────

def index_video(state: VideoAudioState) -> Dict[str, Any]:
    """
    Downloads the video (if YouTube URL) then runs the full local analysis:
      - faster-whisper  → transcript
      - PaddleOCR       → on-screen text
      - BLIP            → visual scene captions

    Returns extracted signals merged into state.
    """
    video_url: str = state.get("video_url", "")
    video_id: str = state.get("video_id", f"vid_{int(time.time())}")

    logger.info("---[Node:Indexer] Processing video: %s", video_url)

    local_path = "temp_audit_video.mp4"
    errors: List[str] = []

    try:
        service = LocalMediaService()

        # ── 1. Download if YouTube URL ──────────────────────────────────────
        if "youtube.com" in video_url or "youtu.be" in video_url:
            logger.info("[Indexer] YouTube URL detected — downloading with yt-dlp…")
            local_path = service.download_youtube_video(video_url, output_path=local_path)
        else:
            local_path = video_url  # treat as local file path

        # ── 2. Run full local analysis ──────────────────────────────────────
        logger.info("[Indexer] Running local multimodal analysis…")
        analysis = service.analyse(local_path)

        logger.info(
            "[Indexer] Done — transcript=%d chars, ocr=%d chars, captions=%d",
            len(analysis["transcript"]),
            len(analysis["ocr_text"]),
            len(analysis["visual_tags"]),
        )

        return {
            "transcript": analysis["transcript"],
            "ocr_text": analysis["ocr_text"],
            "visual_tags": analysis["visual_tags"],
            "duration_seconds": analysis["duration_seconds"],
            "errors": errors,
        }

    except Exception as exc:
        logger.exception("[Indexer] Failed: %s", exc)
        return {
            "transcript": "",
            "ocr_text": "",
            "visual_tags": [],
            "duration_seconds": 0.0,
            "errors": [f"index_video error: {exc}"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 – RAG: Retrieve Compliance Rules from ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_compliance_rules(state: VideoAudioState) -> Dict[str, Any]:
    """
    Builds a query from the video signals and retrieves the most relevant
    compliance rule chunks from ChromaDB using Ollama embeddings.
    """
    logger.info("---[Node:Retriever] Retrieving compliance rules from ChromaDB…")

    transcript = state.get("transcript", "")
    ocr_text = state.get("ocr_text", "")
    visual_tags = state.get("visual_tags", [])

    # Build a compact query string from the multimodal signals
    query = " ".join(filter(None, [
        transcript[:500],
        ocr_text[:300],
        " ".join(visual_tags[:10]),
    ]))

    if not query.strip():
        return {"compliance_rules": "", "errors": ["retrieve: no video signals available"]}

    try:
        collection = _get_vectorstore()
        results = collection.query(query_texts=[query], n_results=8)

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        chunks = []
        for doc, meta in zip(docs, metadatas):
            source = meta.get("source", "unknown") if meta else "unknown"
            chunks.append(f"[Source: {source}]\n{doc}")

        compliance_rules = "\n\n---\n\n".join(chunks)
        logger.info("[Retriever] Retrieved %d rule chunks", len(chunks))

        return {"compliance_rules": compliance_rules}

    except Exception as exc:
        logger.exception("[Retriever] Failed: %s", exc)
        return {
            "compliance_rules": "",
            "errors": [f"retrieve_compliance_rules error: {exc}"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 – Audit: LLM Compliance Evaluation via Ollama
# ─────────────────────────────────────────────────────────────────────────────

_AUDIT_SYSTEM_PROMPT = """You are a senior compliance auditor specialising in digital advertising.
You will be given extracted signals from a video ad (transcript, on-screen text, visual descriptions)
and a set of relevant compliance rules.

Your task: identify every compliance violation. For each issue output a JSON object with:
  - category  (string): e.g. "FTC Disclosure", "Claim Substantiation", "Brand Safety", "Ad Spec"
  - severity  (string): "critical" | "warning" | "info"
  - description (string): clear explanation of the violation
  - timestamp (string|null): video timestamp if detectable, e.g. "00:01:23"
  - remediation (string|null): suggested fix

Return a JSON array of issue objects. If there are no violations return an empty array [].
Output ONLY the JSON array — no prose, no markdown fences."""

_AUDIT_USER_TEMPLATE = """## Compliance Rules
{rules}

## Video Signals

### Spoken Transcript
{transcript}

### On-Screen Text (OCR)
{ocr}

### Visual Scene Descriptions
{visuals}

### Duration
{duration:.1f} seconds

Identify all compliance violations:"""


def audit_compliance(state: VideoAudioState) -> Dict[str, Any]:
    """
    Sends the multimodal signals + retrieved rules to the local Ollama LLM
    and parses the structured list of compliance issues.
    """
    logger.info("---[Node:Audit] Evaluating compliance with Ollama…")

    transcript = state.get("transcript", "")
    ocr_text = state.get("ocr_text", "")
    visual_tags = state.get("visual_tags", [])
    rules = state.get("compliance_rules", "No rules retrieved.")
    duration = state.get("duration_seconds", 0.0)

    user_msg = _AUDIT_USER_TEMPLATE.format(
        rules=rules[:3000],
        transcript=transcript[:2000],
        ocr=ocr_text[:1000],
        visuals="\n".join(visual_tags[:20]),
        duration=duration,
    )

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=_AUDIT_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        raw = response.content.strip()
        # Strip markdown fences if the model adds them anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        issues: List[ComplianceIssue] = json.loads(raw)
        logger.info("[Audit] Detected %d issues", len(issues))
        return {"issues": issues}

    except json.JSONDecodeError as exc:
        logger.warning("[Audit] Could not parse LLM JSON: %s", exc)
        return {"issues": [], "errors": [f"audit JSON parse error: {exc}"]}
    except Exception as exc:
        logger.exception("[Audit] Failed: %s", exc)
        return {"issues": [], "errors": [f"audit_compliance error: {exc}"]}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 – Report Generation
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_DEDUCTIONS = {"critical": 20, "warning": 7, "info": 2}

_REPORT_SYSTEM_PROMPT = """You are a compliance report writer. Given a list of detected violations,
produce a concise executive summary (3–5 sentences) and a prioritised remediation plan.
Return a JSON object with keys:
  - "summary"    (string): executive overview
  - "remediation_plan" (list of strings): ordered action items, most critical first
Output ONLY the JSON object."""


def generate_report(state: VideoAudioState) -> Dict[str, Any]:
    """
    Calculates the compliance score and uses Ollama to write the final
    executive summary and remediation plan.
    """
    logger.info("---[Node:Report] Generating final compliance report…")

    issues: List[ComplianceIssue] = state.get("issues", [])
    video_id: str = state.get("video_id", "unknown")
    duration: float = state.get("duration_seconds", 0.0)
    errors: List[str] = state.get("errors", [])

    # The Azure implementation marked an audit as failed when indexing did not
    # produce a transcript. Keep that error behaviour with the local services:
    # a download/model/indexing failure must never be reported as a clean pass.
    has_video_signals = bool(
        state.get("transcript")
        or state.get("ocr_text")
        or state.get("visual_tags")
    )
    if errors and not has_video_signals:
        report = {
            "video_id": video_id,
            "compliance_score": 0,
            "status": "FAIL",
            "duration_seconds": duration,
            "total_issues": 0,
            "issues_by_severity": {"critical": 0, "warning": 0, "info": 0},
            "issues": [],
            "summary": "Audit skipped because video processing failed.",
            "remediation_plan": [],
        }
        logger.warning("[Report] Marking audit as failed after processing error(s): %s", errors)
        return {"compliance_score": 0, "report": report}

    # ── Score calculation ────────────────────────────────────────────────────
    score = 100
    for issue in issues:
        score -= _SEVERITY_DEDUCTIONS.get(issue.get("severity", "info"), 2)
    score = max(0, score)
    status = "PASS" if score >= 80 else "FAIL"

    # ── LLM-written summary ──────────────────────────────────────────────────
    summary = ""
    remediation_plan: List[str] = []

    if issues:
        try:
            llm = _get_llm()
            issues_str = json.dumps(issues, indent=2)
            response = llm.invoke([
                SystemMessage(content=_REPORT_SYSTEM_PROMPT),
                HumanMessage(content=f"Issues detected:\n{issues_str}"),
            ])
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            report_data = json.loads(raw)
            summary = report_data.get("summary", "")
            remediation_plan = report_data.get("remediation_plan", [])
        except Exception as exc:
            logger.warning("[Report] LLM report generation failed: %s", exc)
            summary = f"Compliance audit detected {len(issues)} issue(s)."
    else:
        summary = "No compliance issues detected. The video asset meets all reviewed guidelines."

    report = {
        "video_id": video_id,
        "compliance_score": score,
        "status": status,
        "duration_seconds": duration,
        "total_issues": len(issues),
        "issues_by_severity": {
            "critical": sum(1 for i in issues if i.get("severity") == "critical"),
            "warning": sum(1 for i in issues if i.get("severity") == "warning"),
            "info": sum(1 for i in issues if i.get("severity") == "info"),
        },
        "issues": issues,
        "summary": summary,
        "remediation_plan": remediation_plan,
    }

    logger.info(
        "[Report] Score=%d  Status=%s  Issues=%d", score, status, len(issues)
    )
    return {"compliance_score": score, "report": report}
