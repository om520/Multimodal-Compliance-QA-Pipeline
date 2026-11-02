"""Persistent, local audit evidence storage."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def save_audit_run(project_root: Path, video_url: str, state: Mapping[str, Any]) -> Path:
    """Save the non-model inputs and outputs needed to reproduce an audit."""
    output_dir = Path(os.getenv("AUDIT_RUNS_DIR", project_root / "audit_runs"))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = str(state.get("video_id", "unknown"))
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "video_id": video_id,
        "video_url": video_url,
        "transcript": state.get("transcript", ""),
        "ocr_text": state.get("ocr_text", ""),
        "visual_tags": state.get("visual_tags", []),
        "duration_seconds": state.get("duration_seconds", 0.0),
        "retrieved_rules": state.get("compliance_rules", ""),
        "issues": state.get("issues", []),
        "report": state.get("report", {}),
        "errors": state.get("errors", []),
    }

    destination = output_dir / f"{video_id}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)
    return destination
