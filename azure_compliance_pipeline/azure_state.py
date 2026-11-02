"""
state.py â€“ Shared LangGraph state definitions for the Compliance QA Pipeline.

All graph nodes communicate exclusively through ``VideoAudioState``.
Annotated fields with ``operator.add`` accumulate values across parallel
branches (e.g. multiple violation checks running concurrently).
"""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class ComplianceIssue(TypedDict):
    """Represents a single compliance violation detected by the audit node."""

    category: str          # e.g. "Disclosure", "Claim Substantiation", "Ad Spec"
    severity: str          # "critical" | "warning" | "info"
    description: str       # Human-readable explanation of the issue
    timestamp: Optional[str]  # Video timestamp where issue occurs, e.g. "00:04:12"
    remediation: Optional[str]  # Suggested fix


class VideoAudioState(TypedDict):
    """
    Central state object threaded through every LangGraph node.

    Fields annotated with ``operator.add`` support parallel fan-out/fan-in:
    each branch appends to the list and the results are merged automatically.
    """

    # â”€â”€ Input â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    video_url: str          # YouTube URL or direct video link
    video_id: str           # Unique run identifier used for Azure VI job IDs

    # â”€â”€ Ingestion / Extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    local_file_path: Optional[str]          # Local path after yt-dlp download
    blob_url: Optional[str]                 # Azure Blob Storage URL of uploaded asset
    video_metadata: Dict[str, Any]          # Raw metadata from Azure Video Indexer
    transcript: Optional[str]              # Full spoken-word transcript
    ocr_text: List[str]                    # On-screen text snippets (timestamped)
    visual_tags: List[str]                 # Scene-level visual labels / keywords
    duration_seconds: Optional[float]      # Total video duration

    # â”€â”€ Knowledge Retrieval â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    retrieved_rules: List[str]             # Policy chunks from Azure AI Search

    # â”€â”€ Compliance Evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    compliance_results: Annotated[List[ComplianceIssue], operator.add]

    # â”€â”€ Output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    final_status: str        # "PASS" | "FAIL" | "PENDING"
    final_report: str        # Full markdown audit report
    audit_score: Optional[float]  # 0.0â€“100.0 compliance score

    # â”€â”€ System / Observability â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    errors: Annotated[List[str], operator.add]  # Non-fatal errors accumulate
