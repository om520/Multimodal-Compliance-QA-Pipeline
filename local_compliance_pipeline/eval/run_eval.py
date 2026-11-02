"""
run_eval.py – Batch evaluation runner for the Compliance QA Pipeline.

This script:
  1. Loads ground_truth.csv (labels committed before any pipeline output is seen)
  2. Injects synthetic transcripts directly into the pipeline state,
     bypassing Azure Video Indexer (so no real videos or Azure credentials needed
     for the indexing step — though Azure OpenAI + Azure Search ARE called for
     the RAG + audit + report nodes)
  3. Invokes compliance_graph for each clip
  4. Saves predictions to predictions.csv

Usage:
    # From the complianceQApipeline/ directory:
    python eval/run_eval.py [--split train|holdout|all] [--output eval/predictions.csv]

    # Dry-run with mock LLM (no Azure calls):
    python eval/run_eval.py --mock

Environment:
    Requires .env with Azure credentials (AZURE_OPENAI_*, AZURE_SEARCH_*).
    Azure Video Indexer credentials are NOT required for synthetic eval.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Path bootstrap ────────────────────────────────────────────────────────────
# _HERE = complianceQApipeline/eval/
# _PKG_ROOT = Desktop/multimodel/  (the directory that CONTAINS complianceQApipeline)
_HERE = Path(__file__).resolve().parent          # complianceQApipeline/eval/
_PKG_ROOT = _HERE.parent.parent                  # Desktop/multimodel/
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from dotenv import load_dotenv
load_dotenv(_PKG_ROOT / "complianceQApipeline" / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval-runner")

GROUND_TRUTH_CSV = _HERE / "ground_truth.csv"
DEFAULT_OUTPUT    = _HERE / "predictions.csv"

# ── Fieldnames for predictions CSV ────────────────────────────────────────────
PRED_FIELDS = [
    "clip_id",
    "split",
    "gt_pass_fail",
    "gt_violation_category",
    "pred_pass_fail",
    "pred_audit_score",
    "pred_violation_categories",   # comma-separated categories from issues list
    "pred_severity_max",           # worst severity among detected issues
    "num_issues",
    "pipeline_errors",
    "latency_seconds",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_ground_truth(csv_path: Path, split: str = "all") -> List[Dict]:
    """Load ground truth rows, optionally filtered by split."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split == "all" or row["split"] == split:
                rows.append(row)
    logger.info("Loaded %d ground truth rows (split=%s)", len(rows), split)
    return rows


def _severity_rank(s: str) -> int:
    return {"critical": 3, "warning": 2, "info": 1}.get(s, 0)


def _extract_categories(issues: List[Dict]) -> str:
    """Return unique violation categories from issues as a comma-separated string."""
    seen = []
    for issue in issues:
        cat = issue.get("category", "Unknown")
        if cat not in seen:
            seen.append(cat)
    return "; ".join(seen) if seen else ""


def _extract_max_severity(issues: List[Dict]) -> str:
    if not issues:
        return "none"
    ranked = sorted(issues, key=lambda i: _severity_rank(i.get("severity", "info")), reverse=True)
    return ranked[0].get("severity", "info")


# ── Mock mode (no Azure calls) ────────────────────────────────────────────────

def _mock_invoke(clip_id: str, clip_data: Dict, gt_row: Dict) -> Dict[str, Any]:
    """
    Simulate pipeline output without calling any Azure service.
    Useful for testing the eval harness itself before real API calls.

    Logic: return the ground truth answer with small controlled noise to simulate
    realistic model behavior (~85% accuracy).
    """
    import random
    random.seed(hash(clip_id) % (2**31))

    gt_verdict = gt_row["gt_pass_fail"]
    gt_cat     = gt_row["gt_violation_category"]

    # 85% of the time predict correctly, 15% flip the verdict
    correct = random.random() < 0.85

    if correct and gt_verdict == "FAIL":
        pred_verdict = "FAIL"
        issues = [{"category": gt_cat, "severity": gt_row["gt_severity"],
                   "description": f"Simulated detection of {gt_cat} issue.",
                   "timestamp": None, "remediation": "See policy."}]
        score = 60.0
    elif correct and gt_verdict == "PASS":
        pred_verdict = "PASS"
        issues = []
        score = 95.0
    elif not correct and gt_verdict == "FAIL":
        # False negative
        pred_verdict = "PASS"
        issues = []
        score = 88.0
    else:
        # False positive
        pred_verdict = "FAIL"
        issues = [{"category": "FTC Disclosure", "severity": "warning",
                   "description": "Simulated false positive disclosure issue.",
                   "timestamp": None, "remediation": "Review disclosure placement."}]
        score = 72.0

    return {
        "final_status": pred_verdict,
        "audit_score": score,
        "compliance_results": issues,
        "errors": [],
    }


# ── Real pipeline invocation ──────────────────────────────────────────────────

def _real_invoke(clip_id: str, clip_data: Dict, video_url: str) -> Dict[str, Any]:
    """
    If video_url is SYNTHETIC, inject synthetic clip data directly into the pipeline state,
    bypassing the index_video node.
    If video_url is a real URL, invoke the full compiled LangGraph pipeline.
    """
    if video_url != "SYNTHETIC":
        # Run the full real pipeline (downloads video, uses Azure Video Indexer)
        from complianceQApipeline.backend.src.graph.workflow import compliance_graph
        initial_state = {"video_url": video_url}
        final_state = compliance_graph.invoke(initial_state)
        return final_state

    # Otherwise, do the synthetic bypass
    from complianceQApipeline.backend.src.graph.nodes import (
        retrieve_compliance_rules,
        audit_compliance,
        generate_report,
    )

    # Build state as if index_video already ran
    state: Dict[str, Any] = {
        "video_url": f"SYNTHETIC://{clip_id}",
        "video_id": clip_id,
        "local_file_path": None,
        "blob_url": None,
        "video_metadata": {},
        "transcript": clip_data.get("transcript", ""),
        "ocr_text": clip_data.get("ocr_text", []),
        "visual_tags": clip_data.get("visual_tags", []),
        "duration_seconds": clip_data.get("duration_seconds", 0.0),
        "retrieved_rules": [],
        "compliance_results": [],
        "errors": [],
        "final_status": "PENDING",
        "final_report": "",
        "audit_score": None,
    }

    # Node 2: RAG retrieval
    state.update(retrieve_compliance_rules(state))

    # Node 3: LLM audit
    state.update(audit_compliance(state))

    # Node 4: Report generation
    state.update(generate_report(state))

    return state


# ── Core batch runner ─────────────────────────────────────────────────────────

def run_evaluation(
    split: str = "train",
    output_path: Path = DEFAULT_OUTPUT,
    mock: bool = False,
    delay_seconds: float = 2.0,
) -> Path:
    """
    Run the full batch evaluation and write predictions.csv.

    Args:
        split:         "train", "holdout", or "all"
        output_path:   Where to write the predictions CSV
        mock:          If True, use mock LLM instead of real Azure calls
        delay_seconds: Seconds to wait between clips (rate limiting)

    Returns:
        Path to the written predictions CSV
    """
    # Late import to avoid loading the graph at module level (costs tokens)
    from complianceQApipeline.eval.synthetic_transcripts import CLIP_TRANSCRIPTS

    gt_rows = load_ground_truth(GROUND_TRUTH_CSV, split=split)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(gt_rows)

    for idx, gt_row in enumerate(gt_rows, start=1):
        clip_id = gt_row["clip_id"]
        logger.info("─── [%d/%d] Processing %s (split=%s, gt=%s) ───",
                    idx, total, clip_id, gt_row["split"], gt_row["gt_pass_fail"])

        clip_data = CLIP_TRANSCRIPTS.get(clip_id, {})
        if not clip_data:
            logger.warning("No synthetic transcript found for %s — skipping.", clip_id)
            continue

        t0 = time.perf_counter()
        try:
            if mock:
                result = _mock_invoke(clip_id, clip_data, gt_row)
            else:
                result = _real_invoke(clip_id, clip_data, gt_row["video_url"])
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", clip_id, exc, exc_info=True)
            result = {
                "final_status": "ERROR",
                "audit_score": None,
                "compliance_results": [],
                "errors": [str(exc)],
            }
        latency = time.perf_counter() - t0

        issues: List[Dict] = result.get("compliance_results") or []
        pred_verdict = result.get("final_status", "ERROR")
        # Normalise: pipeline may return "PASS"/"FAIL"/"PENDING"/"ERROR"
        if pred_verdict not in ("PASS", "FAIL"):
            pred_verdict = "FAIL"  # treat error/pending conservatively

        row = {
            "clip_id": clip_id,
            "split": gt_row["split"],
            "gt_pass_fail": gt_row["gt_pass_fail"],
            "gt_violation_category": gt_row["gt_violation_category"],
            "pred_pass_fail": pred_verdict,
            "pred_audit_score": result.get("audit_score"),
            "pred_violation_categories": _extract_categories(issues),
            "pred_severity_max": _extract_max_severity(issues),
            "num_issues": len(issues),
            "pipeline_errors": "; ".join(result.get("errors") or []),
            "latency_seconds": f"{latency:.2f}",
        }
        results.append(row)

        logger.info(
            "  ✓ %s | gt=%s pred=%s score=%.1f issues=%d latency=%.1fs",
            clip_id,
            gt_row["gt_pass_fail"],
            pred_verdict,
            result.get("audit_score") or 0.0,
            len(issues),
            latency,
        )

        # Rate-limit real API calls
        if not mock and idx < total:
            time.sleep(delay_seconds)

    # Write predictions CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PRED_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    logger.info("✅  Predictions written to: %s (%d rows)", output_path, len(results))
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch evaluator for the Compliance QA Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--split",
        choices=["train", "holdout", "all"],
        default="train",
        help="Which subset of clips to run (default: train — never run holdout until final eval).",
    )
    p.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Path for predictions CSV (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM responses instead of real Azure API calls (for testing the harness).",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between API calls to respect rate limits (default: 2.0).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.mock:
        logger.info("🔵 Running in MOCK mode — no Azure API calls will be made.")
    else:
        logger.info("🟢 Running in REAL mode — Azure OpenAI + Azure Search will be called.")
        logger.warning(
            "⚠️  Never run --split holdout until you have FINALISED the pipeline. "
            "Holdout is for the single final reported number only."
        )

    out = run_evaluation(
        split=args.split,
        output_path=Path(args.output),
        mock=args.mock,
        delay_seconds=args.delay,
    )
    logger.info("Run complete. Now run:  python eval/metrics.py --predictions %s", out)
