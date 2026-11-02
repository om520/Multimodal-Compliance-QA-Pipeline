"""
state.py – Shared LangGraph state definitions for the Compliance QA Pipeline.

All graph nodes communicate exclusively through ``VideoAudioState``.
Annotated fields with ``operator.add`` accumulate values across parallel
branches (e.g. multiple violation checks running concurrently).
"""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class ComplianceIssue(TypedDict):
    """Represents a single compliance violation detected by the audit node."""

    category: str            # e.g. "Disclosure", "Claim Substantiation", "Ad Spec"
    severity: str            # "critical" | "warning" | "info"
    description: str         # Human-readable explanation of the issue
    timestamp: Optional[str] # Video timestamp where issue occurs, e.g. "00:04:12"
    remediation: Optional[str]  # Suggested fix


class VideoAudioState(TypedDict):
    """
    Central state object threaded through every LangGraph node.

    Fields annotated with ``operator.add`` support parallel fan-out/fan-in:
    each branch appends to the list and the results are merged automatically.
    """

    # ── Input ─────────────────────────────────────────────────────────────────
    video_url: str               # YouTube URL or local file path
    video_id: str                # Unique job identifier

    # ── Extracted multimodal signals ──────────────────────────────────────────
    transcript: str              # Full spoken transcript (faster-whisper)
    ocr_text: str                # On-screen text extracted via PaddleOCR
    visual_tags: List[str]       # Scene/object descriptions from BLIP
    duration_seconds: float      # Video duration in seconds

    # ── RAG context ───────────────────────────────────────────────────────────
    compliance_rules: str        # Relevant rule chunks retrieved from ChromaDB

    # ── Audit results ─────────────────────────────────────────────────────────
    issues: Annotated[List[ComplianceIssue], operator.add]
    compliance_score: int        # 0–100

    # ── Output ────────────────────────────────────────────────────────────────
    report: Dict[str, Any]       # Final structured audit report
    errors: Annotated[List[str], operator.add]  # Non-fatal errors during processing
