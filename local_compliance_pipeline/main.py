"""
main.py – CLI entry point for the local Compliance QA Pipeline.

Usage:
    python main.py --video-url "https://www.youtube.com/watch?v=..." [--video-id my_id]
    python main.py --video-url "/local/path/to/video.mp4"

Runs the full LangGraph pipeline synchronously and prints the audit report
to stdout. Useful for quick local testing without spinning up the FastAPI server.
"""

import argparse
import json
import logging
import sys
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("brand-guardian")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal Compliance QA Pipeline – Local CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video-url",
        required=True,
        help="YouTube URL or local video file path to audit.",
    )
    parser.add_argument(
        "--video-id",
        default=None,
        help="Optional unique identifier for this audit run (default: auto-generated).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_id = args.video_id or f"vid_{uuid.uuid4().hex[:8]}"

    logger.info("Starting compliance audit — video_id=%s", video_id)
    logger.info("Video URL : %s", args.video_url)

    # Import here so logging is configured before heavy model imports
    from backend.src.graph.workflow import compliance_graph

    initial_state = {
        "video_url": args.video_url,
        "video_id": video_id,
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
        result = compliance_graph.invoke(initial_state)
    except Exception as exc:
        # Preserve the original CLI error behaviour: retain the traceback for
        # callers instead of converting a workflow exception into a silent exit.
        logger.exception("Pipeline failed: %s", exc)
        raise

    report = result.get("report", {})
    errors = result.get("errors", [])

    print("\n" + "=" * 60)
    print("  COMPLIANCE AUDIT REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))

    if errors:
        print("\n⚠️  Non-fatal errors during processing:")
        for err in errors:
            print(f"  - {err}")

    score = report.get("compliance_score", 0)
    status = report.get("status", "UNKNOWN")
    print(f"\n{'✅' if status == 'PASS' else '❌'}  Score: {score}/100  |  Status: {status}")
    print("=" * 60)

    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
