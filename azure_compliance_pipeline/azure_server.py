"""
server.py â€“ FastAPI application for the Compliance QA Pipeline.

Endpoints:
  POST /audit          â€“ Submit a video URL for compliance auditing (async job)
  GET  /audit/{job_id} â€“ Poll the status / result of a submitted job
  GET  /health         â€“ Liveness probe

Jobs are executed in a background thread pool so the HTTP response is
returned immediately. Results are cached in an in-memory dict (swap for
Redis in production via the redis dependency already in pyproject.toml).
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from backend.src.api.telemetry import setup_telemetry
from backend.src.graph.workflow import compliance_graph

load_dotenv()

# â”€â”€ Telemetry must be set up before the app handles any requests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setup_telemetry()

logger = logging.getLogger("brand-guardian")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="Multimodal Compliance QA Pipeline",
    description=(
        "Automated brand compliance auditing for video assets. "
        "Powered by Azure Video Indexer, Azure AI Search, and LangGraph."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# â”€â”€ In-memory job store (replace with Redis for production) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_jobs: Dict[str, Dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=4)


# â”€â”€ Request / Response Models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class AuditRequest(BaseModel):
    video_url: str
    video_id: Optional[str] = None


class AuditJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class AuditResultResponse(BaseModel):
    job_id: str
    status: str              # "pending" | "running" | "complete" | "error"
    audit_score: Optional[float] = None
    final_status: Optional[str] = None
    final_report: Optional[str] = None
    compliance_issues: Optional[list] = None
    errors: Optional[list] = None


# â”€â”€ Background worker â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _run_audit(job_id: str, video_url: str, video_id: str) -> None:
    """Execute the LangGraph pipeline in a background thread."""
    _jobs[job_id]["status"] = "running"
    logger.info("[Server] Starting audit job %s for %s", job_id, video_url)

    initial_state = {
        "video_url": video_url,
        "video_id": video_id,
        "ocr_text": [],
        "visual_tags": [],
        "retrieved_rules": [],
        "compliance_results": [],
        "errors": [],
        "video_metadata": {},
    }

    try:
        result = compliance_graph.invoke(initial_state)
        _jobs[job_id].update(
            {
                "status": "complete",
                "audit_score": result.get("audit_score"),
                "final_status": result.get("final_status"),
                "final_report": result.get("final_report"),
                "compliance_issues": result.get("compliance_results", []),
                "errors": result.get("errors", []),
            }
        )
        logger.info("[Server] Job %s complete. Score=%.1f", job_id, result.get("audit_score", 0))
    except Exception as exc:
        logger.error("[Server] Job %s failed: %s", job_id, exc, exc_info=True)
        _jobs[job_id].update({"status": "error", "errors": [str(exc)]})


# â”€â”€ Routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/health", tags=["System"])
def health_check() -> Dict[str, str]:
    """Liveness probe for load balancers and container orchestrators."""
    return {"status": "ok", "service": "compliance-qa-pipeline"}


@app.post("/audit", response_model=AuditJobResponse, status_code=202, tags=["Audit"])
def submit_audit(request: AuditRequest) -> AuditJobResponse:
    """
    Submit a video URL for compliance auditing.

    Returns a ``job_id`` immediately. Poll ``GET /audit/{job_id}`` for results.
    """
    job_id = str(uuid.uuid4())
    video_id = request.video_id or f"vid_{job_id[:8]}"

    _jobs[job_id] = {"status": "pending", "video_url": request.video_url}

    _executor.submit(_run_audit, job_id, request.video_url, video_id)

    logger.info("[Server] Accepted audit job %s", job_id)
    return AuditJobResponse(
        job_id=job_id,
        status="pending",
        message=f"Audit job submitted. Poll GET /audit/{job_id} for results.",
    )


@app.get("/audit/{job_id}", response_model=AuditResultResponse, tags=["Audit"])
def get_audit_result(job_id: str) -> AuditResultResponse:
    """
    Retrieve the status and results of a compliance audit job.

    Status values: ``pending`` â†’ ``running`` â†’ ``complete`` | ``error``
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return AuditResultResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        audit_score=job.get("audit_score"),
        final_status=job.get("final_status"),
        final_report=job.get("final_report"),
        compliance_issues=job.get("compliance_issues"),
        errors=job.get("errors"),
    )
