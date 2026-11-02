"""FastAPI API for the local Compliance QA Pipeline."""

import logging
import shutil
import uuid
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse

# Load configuration before importing the graph, whose local services use it.
load_dotenv(override=True)

from backend.src.api.telemetry import setup_telemetry
from backend.src.graph.workflow import compliance_graph
from backend.src.services.audit_store import save_audit_run
from backend.src.services.component_status import component_status

setup_telemetry()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api-server")

app = FastAPI(
    title="Brand Guardian AI API",
    description="API for auditing video content against brand compliance rules.",
    version="1.0.0",
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_INDEX = PROJECT_ROOT / "frontend" / "index.html"
UPLOAD_DIR = PROJECT_ROOT / "local_uploads"


# Keep the established public Pydantic request/response schema intact.
class AuditRequest(BaseModel):
    video_url: str


class ComplianceIssue(BaseModel):
    category: str
    severity: str
    description: str


class AuditResponse(BaseModel):
    session_id: str
    video_id: str
    status: str
    final_report: str
    compliance_results: List[ComplianceIssue]


@app.post("/audit", response_model=AuditResponse)
async def audit_video(request: AuditRequest):
    """Run one local compliance audit and return the established response shape."""
    session_id = str(uuid.uuid4())
    video_id_short = f"vid_{session_id[:8]}"
    logger.info("Received Audit Request: %s (Session: %s)", request.video_url, session_id)

    # These additional fields are required by the local graph, but do not alter
    # the API contract or Pydantic validation above.
    initial_inputs = {
        "video_url": request.video_url,
        "video_id": video_id_short,
        "transcript": "",
        "ocr_text": "",
        "visual_tags": [],
        "duration_seconds": 0.0,
        "compliance_rules": "",
        "issues": [],
        "compliance_score": 100,
        "report": {},
        "errors": [],
    }

    try:
        # Video analysis and local model inference are blocking operations.
        # Keep them off FastAPI's event loop while the browser waits for its
        # synchronous response.
        final_state = await run_in_threadpool(compliance_graph.invoke, initial_inputs)
        report = final_state.get("report", {})
        try:
            evidence_path = save_audit_run(PROJECT_ROOT, request.video_url, final_state)
            logger.info("Saved audit evidence to %s", evidence_path)
        except OSError as exc:
            # Persistence must not hide an otherwise valid audit result.
            logger.warning("Could not save audit evidence: %s", exc)
        return AuditResponse(
            session_id=session_id,
            video_id=final_state.get("video_id", video_id_short),
            status=report.get("status", "UNKNOWN"),
            final_report=report.get("summary", "No report generated."),
            compliance_results=final_state.get("issues", []),
        )
    except Exception as exc:
        # Preserve the original failure behaviour: log it and return HTTP 500.
        logger.error("Audit Failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Workflow Execution Failed: {exc}",
        ) from exc


@app.post("/audit/upload", response_model=AuditResponse)
async def upload_and_audit(video: UploadFile = File(...)):
    """Store an uploaded local video, then audit it using the same API contract."""
    if not video.filename:
        raise HTTPException(status_code=400, detail="Please choose a video file.")
    if video.content_type and not video.content_type.startswith("video/"):
        raise HTTPException(status_code=415, detail="Uploaded file must be a video.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(video.filename).suffix.lower() or ".mp4"
    destination = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    try:
        with destination.open("wb") as output_file:
            shutil.copyfileobj(video.file, output_file)
    except OSError as exc:
        logger.error("Could not save uploaded video: %s", exc)
        raise HTTPException(status_code=500, detail="Could not save the uploaded video.") from exc
    finally:
        await video.close()

    return await audit_video(AuditRequest(video_url=str(destination)))


@app.get("/", include_in_schema=False)
def browser_app():
    """Serve the bundled browser interface from the same origin as the API."""
    if not FRONTEND_INDEX.is_file():
        raise HTTPException(status_code=404, detail="Browser interface was not found.")
    return FileResponse(FRONTEND_INDEX)


@app.get("/health")
def health_check():
    """Simple liveness endpoint."""
    return {"status": "healthy", "service": "Brand Guardian AI"}


@app.get("/health/components")
def health_components():
    """Report the readiness of each local model and database component."""
    return component_status(PROJECT_ROOT)
