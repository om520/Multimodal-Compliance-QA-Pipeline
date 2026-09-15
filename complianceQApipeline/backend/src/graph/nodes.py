"""
nodes.py – LangGraph node implementations for the Compliance QA Pipeline.

Each function receives the shared ``VideoAudioState`` dict and returns a
partial update dict that LangGraph merges back into the state.

Node execution order (defined in workflow.py):
  index_video → retrieve_compliance_rules → audit_compliance → generate_report
"""

import json
import logging
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from backend.src.graph.state import ComplianceIssue, VideoAudioState
from backend.src.services.video_indexer import VideoIndexerService

load_dotenv()

logger = logging.getLogger("brand-guardian")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Shared Azure OpenAI client (reused across nodes)
# ---------------------------------------------------------------------------

def _get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        temperature=0,
    )


def _get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI__EMBEDDING_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    )


# ---------------------------------------------------------------------------
# Node 1 – Video Ingestion & Indexing
# ---------------------------------------------------------------------------

def index_video(state: VideoAudioState) -> Dict[str, Any]:
    """
    Downloads the video (if YouTube), uploads it to Azure Blob Storage,
    submits it to Azure Video Indexer, and waits for insights.

    Returns extracted transcript, OCR text, visual tags, and metadata.
    """
    video_url: str = state.get("video_url", "")
    video_id: str = state.get("video_id", "vid_demo")

    logger.info("---[Node:Indexer] Processing video from url: %s", video_url)

    local_filename = "temp_audit_video.mp4"
    errors: List[str] = []

    try:
        vi_service = VideoIndexerService()

        # ── 1. Download YouTube video locally via yt-dlp ──
        if "youtube.com" in video_url or "youtu.be" in video_url:
            logger.info("[Indexer] Detected YouTube URL — downloading with yt-dlp…")
            local_path = vi_service.download_youtube_video(video_url, output_path=local_filename)
        else:
            local_path = video_url  # assume local path or direct URL

        # ── 2. Upload to Azure Blob Storage and get a SAS URL ──
        logger.info("[Indexer] Uploading video to Azure Blob Storage…")
        blob_url = vi_service.upload_to_blob(local_path, blob_name=f"{video_id}.mp4")

        # ── 3. Submit to Azure Video Indexer and poll for completion ──
        logger.info("[Indexer] Submitting to Azure Video Indexer…")
        vi_video_id = vi_service.submit_video(video_id=video_id, video_url=blob_url, video_name=video_id)
        insights = vi_service.wait_for_indexing(vi_video_id)

        # ── 4. Extract structured features ──
        transcript = vi_service.extract_transcript(insights)
        ocr_text = vi_service.extract_ocr(insights)
        visual_tags = vi_service.extract_visual_tags(insights)
        duration = vi_service.extract_duration(insights)

        logger.info(
            "[Indexer] Extraction complete. Transcript chars=%d, OCR snippets=%d, Tags=%d",
            len(transcript or ""),
            len(ocr_text),
            len(visual_tags),
        )

        return {
            "local_file_path": local_path,
            "blob_url": blob_url,
            "video_metadata": insights,
            "transcript": transcript,
            "ocr_text": ocr_text,
            "visual_tags": visual_tags,
            "duration_seconds": duration,
            "errors": errors,
        }

    except Exception as exc:
        logger.error("[Indexer] Failed: %s", exc, exc_info=True)
        return {
            "transcript": None,
            "ocr_text": [],
            "visual_tags": [],
            "video_metadata": {},
            "errors": [f"index_video error: {exc}"],
        }


# ---------------------------------------------------------------------------
# Node 2 – Compliance Rule Retrieval (RAG)
# ---------------------------------------------------------------------------

def retrieve_compliance_rules(state: VideoAudioState) -> Dict[str, Any]:
    """
    Builds a search query from the transcript + OCR text and retrieves
    relevant compliance rule chunks from Azure AI Search.
    """
    from langchain_community.vectorstores import AzureSearch

    logger.info("---[Node:Retriever] Retrieving relevant compliance rules…")

    transcript: str = state.get("transcript") or ""
    ocr_snippets: List[str] = state.get("ocr_text") or []

    # Build a combined query – truncate to keep token cost manageable
    combined_content = f"{transcript[:2000]}\n\nOn-screen text:\n" + "\n".join(ocr_snippets[:20])

    try:
        embeddings = _get_embeddings()
        vector_store = AzureSearch(
            azure_search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            azure_search_key=os.environ["AZURE_SEARCH_API_KEY"],
            index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
            embedding_function=embeddings.embed_query,
        )

        docs = vector_store.similarity_search(combined_content, k=8)
        rules = [doc.page_content for doc in docs]

        logger.info("[Retriever] Retrieved %d rule chunks.", len(rules))
        return {"retrieved_rules": rules, "errors": []}

    except Exception as exc:
        logger.error("[Retriever] Failed: %s", exc, exc_info=True)
        return {"retrieved_rules": [], "errors": [f"retrieve_compliance_rules error: {exc}"]}


# ---------------------------------------------------------------------------
# Node 3 – Compliance Auditing (LLM evaluation)
# ---------------------------------------------------------------------------

_AUDIT_SYSTEM_PROMPT = """
You are a senior brand compliance auditor. You will be given:
1. A video transcript and on-screen text extracted from a marketing asset.
2. Relevant excerpts from compliance policy manuals.

Your task is to evaluate the asset for the following compliance dimensions:
- **FTC Disclosure**: Are sponsorship/paid partnership disclosures clear, conspicuous, and timely?
- **Claim Substantiation**: Are any performance or benefit claims made that lack substantiation?
- **Brand Safety**: Is there content that could harm brand reputation?
- **Platform Ad Specs**: Do any visual cues violate platform-specific ad specifications?

For each violation found, return a JSON array of objects with these exact keys:
  category, severity (critical|warning|info), description, timestamp (or null), remediation

Return ONLY the JSON array — no markdown, no extra text.
If there are no violations, return an empty array: []
""".strip()


def audit_compliance(state: VideoAudioState) -> Dict[str, Any]:
    """
    Uses Azure OpenAI to evaluate the extracted transcript + OCR against
    retrieved compliance rules and produce a structured list of violations.
    """
    logger.info("---[Node:Auditor] Running compliance audit…")

    transcript: str = state.get("transcript") or "(no transcript available)"
    ocr_text: List[str] = state.get("ocr_text") or []
    rules: List[str] = state.get("retrieved_rules") or []

    rules_block = "\n\n".join(f"[Rule {i+1}]\n{r}" for i, r in enumerate(rules))
    ocr_block = "\n".join(ocr_text) or "(no on-screen text)"

    user_prompt = (
        f"TRANSCRIPT:\n{transcript[:4000]}\n\n"
        f"ON-SCREEN TEXT:\n{ocr_block[:1000]}\n\n"
        f"COMPLIANCE RULES:\n{rules_block[:3000]}"
    )

    try:
        llm = _get_llm()
        messages = [
            SystemMessage(content=_AUDIT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        raw_json = response.content.strip()

        # Strip accidental markdown code fences
        raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json)
        raw_json = re.sub(r"\s*```$", "", raw_json)

        issues_data: List[Dict] = json.loads(raw_json)
        compliance_results: List[ComplianceIssue] = [
            ComplianceIssue(  # type: ignore[call-arg]
                category=item.get("category", "Unknown"),
                severity=item.get("severity", "info"),
                description=item.get("description", ""),
                timestamp=item.get("timestamp"),
                remediation=item.get("remediation"),
            )
            for item in issues_data
        ]

        logger.info("[Auditor] Detected %d compliance issue(s).", len(compliance_results))
        return {"compliance_results": compliance_results, "errors": []}

    except json.JSONDecodeError as exc:
        logger.error("[Auditor] Could not parse LLM response as JSON: %s", exc)
        return {"compliance_results": [], "errors": [f"audit_compliance JSON parse error: {exc}"]}
    except Exception as exc:
        logger.error("[Auditor] Failed: %s", exc, exc_info=True)
        return {"compliance_results": [], "errors": [f"audit_compliance error: {exc}"]}


# ---------------------------------------------------------------------------
# Node 4 – Report Generation
# ---------------------------------------------------------------------------

_REPORT_SYSTEM_PROMPT = """
You are a compliance report writer. Generate a professional, structured markdown
audit report given the violations list and video metadata. Include:
- An executive summary with an overall compliance score (0–100).
- A categorised violation table (category | severity | timestamp | description).
- Actionable remediation steps for each critical/warning issue.
- A final verdict: PASS (score ≥ 80) or FAIL (score < 80).

Return ONLY the markdown report.
""".strip()


def generate_report(state: VideoAudioState) -> Dict[str, Any]:
    """
    Synthesises all compliance results into a final structured audit report
    and computes an overall audit score.
    """
    logger.info("---[Node:Reporter] Generating final compliance report…")

    issues: List[ComplianceIssue] = state.get("compliance_results") or []
    video_url: str = state.get("video_url", "N/A")
    duration: float = state.get("duration_seconds") or 0.0

    # Simple heuristic score: start at 100, deduct per severity
    penalty = {"critical": 20, "warning": 7, "info": 2}
    score = max(0.0, 100.0 - sum(penalty.get(i.get("severity", "info"), 2) for i in issues))

    issues_block = json.dumps(issues, indent=2) if issues else "[]"
    user_prompt = (
        f"Video URL: {video_url}\n"
        f"Duration: {duration:.1f}s\n"
        f"Violations ({len(issues)} total):\n{issues_block}"
    )

    try:
        llm = _get_llm()
        messages = [
            SystemMessage(content=_REPORT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        report_md = response.content.strip()

        final_status = "PASS" if score >= 80.0 else "FAIL"
        logger.info("[Reporter] Report generated. Score=%.1f, Status=%s", score, final_status)

        return {
            "final_report": report_md,
            "audit_score": score,
            "final_status": final_status,
            "errors": [],
        }

    except Exception as exc:
        logger.error("[Reporter] Failed: %s", exc, exc_info=True)
        fallback = f"# Compliance Audit Report\n\n**Score:** {score:.1f}/100\n\n**Issues found:** {len(issues)}\n"
        return {
            "final_report": fallback,
            "audit_score": score,
            "final_status": "PASS" if score >= 80 else "FAIL",
            "errors": [f"generate_report error: {exc}"],
        }
