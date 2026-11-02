"""Run the local pipeline on evaluation videos and save traceable pseudo-gold.

The records are model-generated starting points for human review.  They are
not independent ground truth and must not be used as final accuracy labels
until a reviewer corrects their transcript, OCR, evidence, and violations.

Run from the project root on Windows:
    .\.venv\Scripts\python.exe backend\scripts\build_pseudo_gold.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.src.graph.workflow import compliance_graph
from backend.src.services.audit_store import save_audit_run


VIDEOS = {
    "H9Q0KPUsxbM": "https://www.youtube.com/watch?v=H9Q0KPUsxbM",
    "gOAUXbO9kjg": "https://www.youtube.com/watch?v=gOAUXbO9kjg",
    "V2JJZzIJeUQ": "https://www.youtube.com/watch?v=V2JJZzIJeUQ",
    "gqrGwjP2fU8": "https://www.youtube.com/watch?v=gqrGwjP2fU8",
}


def initial_state(video_id: str, path: Path) -> dict:
    return {
        "video_url": str(path), "video_id": video_id,
        "transcript": "", "ocr_text": "", "visual_tags": [],
        "duration_seconds": 0.0, "compliance_rules": "", "issues": [],
        "compliance_score": 100, "report": {}, "errors": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", action="append", choices=VIDEOS)
    parser.add_argument("--output", default="eval/pseudo_gold/manifest.json")
    args = parser.parse_args()

    selected = args.video_id or list(VIDEOS)
    video_dir = ROOT / "eval" / "videos"
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []

    for video_id in selected:
        path = video_dir / f"{video_id}.mp4"
        if not path.is_file():
            records.append({"video_id": video_id, "error": f"Missing {path}"})
            continue
        logging.info("Processing %s: %s", video_id, path)
        started = time.perf_counter()
        state = initial_state(video_id, path)
        try:
            state = dict(compliance_graph.invoke(state))
        except Exception as exc:  # Save partial output for debugging and review.
            state["errors"] = [*state.get("errors", []), f"pipeline error: {exc}"]
        evidence_path = save_audit_run(ROOT, VIDEOS[video_id], state)
        records.append({
            "video_id": video_id,
            "source_url": VIDEOS[video_id],
            "video_path": str(path.relative_to(ROOT)),
            "evidence_file": str(evidence_path.relative_to(ROOT)),
            "processing_seconds": round(time.perf_counter() - started, 3),
            "status": state.get("report", {}).get("status", "ERROR"),
            "issues": state.get("issues", []),
            "errors": state.get("errors", []),
        })

    output.write_text(json.dumps({
        "dataset_type": "unreviewed_pseudo_gold",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "review_required": True,
        "records": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Wrote manifest to %s", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
