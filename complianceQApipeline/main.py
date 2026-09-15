"""
main.py – CLI entry point for the Compliance QA Pipeline.

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("brand-guardian")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal Compliance QA Pipeline – CLI Trigger",
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

    import uuid
    from dotenv import load_dotenv
    load_dotenv()

    from backend.src.graph.workflow import compliance_graph

    video_id = args.video_id or f"cli_{uuid.uuid4().hex[:8]}"
    logger.info("Starting compliance audit for: %s  (id=%s)", args.video_url, video_id)

    initial_state = {
        "video_url": args.video_url,
        "video_id": video_id,
        "ocr_text": [],
        "visual_tags": [],
        "retrieved_rules": [],
        "compliance_results": [],
        "errors": [],
        "video_metadata": {},
    }

    result = compliance_graph.invoke(initial_state)

    print("\n" + "=" * 70)
    print(f"  COMPLIANCE AUDIT COMPLETE")
    print(f"  Score : {result.get('audit_score', 'N/A')}/100")
    print(f"  Status: {result.get('final_status', 'N/A')}")
    print("=" * 70 + "\n")
    print(result.get("final_report", "(no report generated)"))

    issues = result.get("compliance_results", [])
    if issues:
        print(f"\n--- {len(issues)} Compliance Issue(s) ---")
        print(json.dumps(issues, indent=2))

    errors = result.get("errors", [])
    if errors:
        print(f"\n--- {len(errors)} Non-fatal Error(s) ---")
        for e in errors:
            print(f"  • {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
