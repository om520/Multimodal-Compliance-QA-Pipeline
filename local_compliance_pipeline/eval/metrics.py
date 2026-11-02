"""
metrics.py – Compute evaluation metrics for the Compliance QA Pipeline.

Reads predictions.csv (output of run_eval.py) and ground_truth.csv,
joins them on clip_id, and computes:

  1. Overall pass/fail accuracy (train set)
  2. Per-category precision / recall / F1 (train set)
  3. Holdout pass/fail accuracy (only when --include-holdout is set)
  4. Confusion matrix
  5. A formatted report string suitable for pasting into a README or CV

Usage:
    python eval/metrics.py
    python eval/metrics.py --predictions eval/predictions.csv
    python eval/metrics.py --include-holdout   # only after finalising the pipeline!
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval-metrics")

_HERE = Path(__file__).resolve().parent
GROUND_TRUTH_CSV = _HERE / "ground_truth.csv"
DEFAULT_PREDS    = _HERE / "predictions.csv"

# -- Data loading --------------------------------------------------------------

def load_csv(path: Path) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def join_data(gt_rows: List[Dict], pred_rows: List[Dict]) -> List[Dict]:
    """Inner-join ground truth and predictions on clip_id."""
    pred_map = {r["clip_id"]: r for r in pred_rows}
    joined = []
    for gt in gt_rows:
        cid = gt["clip_id"]
        if cid in pred_map:
            joined.append({**gt, **pred_map[cid]})
        else:
            logger.warning("No prediction found for %s — excluded from metrics.", cid)
    return joined


# -- Core metrics --------------------------------------------------------------

def pass_fail_accuracy(rows: List[Dict]) -> Tuple[float, int, int]:
    """Return (accuracy, n_correct, n_total)."""
    correct = sum(1 for r in rows if r["gt_pass_fail"] == r["pred_pass_fail"])
    return correct / len(rows) if rows else 0.0, correct, len(rows)


def confusion_matrix(rows: List[Dict]) -> Dict[str, int]:
    """Return TP, FP, TN, FN where FAIL=positive class."""
    tp = sum(1 for r in rows if r["gt_pass_fail"] == "FAIL" and r["pred_pass_fail"] == "FAIL")
    fp = sum(1 for r in rows if r["gt_pass_fail"] == "PASS" and r["pred_pass_fail"] == "FAIL")
    tn = sum(1 for r in rows if r["gt_pass_fail"] == "PASS" and r["pred_pass_fail"] == "PASS")
    fn = sum(1 for r in rows if r["gt_pass_fail"] == "FAIL" and r["pred_pass_fail"] == "PASS")
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def per_category_metrics(rows: List[Dict]) -> Dict[str, Dict]:
    """
    Compute per-category precision / recall / F1.

    For category C:
      - TP: gt_violation_category==C and pred correctly flagged FAIL
      - FP: gt_violation_category!=C (or PASS) but pred flagged C
      - FN: gt_violation_category==C and pred missed (PASS or wrong category)

    This treats category detection as a multi-label binary problem per category.
    """
    categories = sorted(set(r["gt_violation_category"] for r in rows))
    results = {}

    for cat in categories:
        tp = fp = fn = 0
        for r in rows:
            gt_cat   = r["gt_violation_category"]
            gt_verd  = r["gt_pass_fail"]
            pred_cats = [c.strip() for c in r.get("pred_violation_categories", "").split(";") if c.strip()]
            pred_verd = r["pred_pass_fail"]

            # TP: ground truth IS this category AND model detected it
            if gt_cat == cat and gt_verd == "FAIL" and cat in pred_cats:
                tp += 1
            # FN: ground truth IS this category AND model missed it
            elif gt_cat == cat and gt_verd == "FAIL" and cat not in pred_cats:
                fn += 1
            # FP: ground truth is NOT this category but model flagged it
            if gt_cat != cat and cat in pred_cats:
                fp += 1

        prec, rec, f1 = precision_recall_f1(tp, fp, fn)
        results[cat] = {
            "TP": tp, "FP": fp, "FN": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "support": sum(1 for r in rows if r["gt_violation_category"] == cat and r["gt_pass_fail"] == "FAIL"),
        }

    return results


# -- sklearn-style classification_report (no sklearn dep required) -------------

def classification_report_str(cat_metrics: Dict[str, Dict]) -> str:
    """Format per-category metrics as a classification-report-style table."""
    header = f"{'Category':<26} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>8}"
    sep    = "-" * len(header)
    lines  = [header, sep]

    total_tp = total_fp = total_fn = 0
    for cat, m in sorted(cat_metrics.items()):
        lines.append(
            f"{cat:<26} {m['precision']:>10.2f} {m['recall']:>10.2f} {m['f1']:>8.2f} {m['support']:>8d}"
        )
        total_tp += m["TP"]
        total_fp += m["FP"]
        total_fn += m["FN"]

    macro_p = sum(m["precision"] for m in cat_metrics.values()) / len(cat_metrics)
    macro_r = sum(m["recall"]    for m in cat_metrics.values()) / len(cat_metrics)
    macro_f = sum(m["f1"]        for m in cat_metrics.values()) / len(cat_metrics)
    micro_p, micro_r, micro_f1 = precision_recall_f1(total_tp, total_fp, total_fn)

    lines += [
        sep,
        f"{'macro avg':<26} {macro_p:>10.2f} {macro_r:>10.2f} {macro_f:>8.2f}",
        f"{'micro avg':<26} {micro_p:>10.2f} {micro_r:>10.2f} {micro_f1:>8.2f}",
    ]
    return "\n".join(lines)


def confusion_matrix_str(cm: Dict[str, int]) -> str:
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    return (
        "Confusion Matrix (positive class = FAIL):\n"
        f"                Pred FAIL   Pred PASS\n"
        f"  GT FAIL     {tp:>8d}  {fn:>10d}\n"
        f"  GT PASS     {fp:>8d}  {tn:>10d}\n"
    )


# -- Report builder ------------------------------------------------------------

def build_report(
    train_rows: List[Dict],
    holdout_rows: Optional[List[Dict]],
    include_holdout: bool,
) -> str:
    """Build the full text report for stdout and README insertion."""
    lines = []
    lines.append("=" * 65)
    lines.append("  COMPLIANCE QA PIPELINE — EVALUATION REPORT")
    lines.append("=" * 65)

    # -- Training set ----------------------------------------------
    acc, correct, n = pass_fail_accuracy(train_rows)
    cm = confusion_matrix(train_rows)
    cat_metrics = per_category_metrics(train_rows)

    lines.append(f"\n## Training Set  (N={n})")
    lines.append(f"Pass/Fail Accuracy : {acc:.1%}  ({correct}/{n})")
    lines.append("")
    lines.append(confusion_matrix_str(cm))
    lines.append("")
    lines.append("Per-Category Metrics:")
    lines.append(classification_report_str(cat_metrics))

    # -- Holdout set -----------------------------------------------
    if holdout_rows and include_holdout:
        h_acc, h_correct, h_n = pass_fail_accuracy(holdout_rows)
        h_cm = confusion_matrix(holdout_rows)
        h_cat = per_category_metrics(holdout_rows)

        lines.append(f"\n## Holdout Set  (N={h_n})  <-  FINAL REPORTED NUMBER")
        lines.append(f"Pass/Fail Accuracy : {h_acc:.1%}  ({h_correct}/{h_n})")
        lines.append("")
        lines.append(confusion_matrix_str(h_cm))
        lines.append("")
        lines.append("Per-Category Metrics:")
        lines.append(classification_report_str(h_cat))

        # -- Combined defensible quote -----------------------------
        total_n = n + h_n
        cats_str = ", ".join(sorted(set(r["gt_violation_category"] for r in train_rows + holdout_rows)))
        lines.append("\n" + "-" * 65)
        lines.append("DEFENSIBLE RESULT STRING (paste into CV / README):")
        lines.append("-" * 65)
        lines.append(
            f'"{h_acc:.0%} pass/fail accuracy on {h_n} held-out, self-labeled clips '
            f"({total_n} total spanning {cats_str}) "
            f'using ground-truth labels committed before any pipeline output was seen."'
        )
    elif not include_holdout:
        lines.append("\n[!]️  Holdout set NOT included (run with --include-holdout only after finalising pipeline).")

    lines.append("\n" + "=" * 65)
    return "\n".join(lines)


# -- CLI -----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute evaluation metrics for the Compliance QA Pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--predictions", default=str(DEFAULT_PREDS),
                   help=f"Path to predictions CSV (default: {DEFAULT_PREDS})")
    p.add_argument("--ground-truth", default=str(GROUND_TRUTH_CSV),
                   help=f"Path to ground truth CSV (default: {GROUND_TRUTH_CSV})")
    p.add_argument("--include-holdout", action="store_true",
                   help="Include holdout metrics. ONLY use this for the final reported number.")
    p.add_argument("--output-json", default=None,
                   help="Optionally write metrics as JSON to this path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    gt_path   = Path(args.ground_truth)
    pred_path = Path(args.predictions)

    if not gt_path.exists():
        logger.error("Ground truth file not found: %s", gt_path)
        sys.exit(1)
    if not pred_path.exists():
        logger.error(
            "Predictions file not found: %s\nRun:  python eval/run_eval.py  first.", pred_path
        )
        sys.exit(1)

    gt_rows   = load_csv(gt_path)
    pred_rows = load_csv(pred_path)

    all_joined    = join_data(gt_rows, pred_rows)
    train_joined  = [r for r in all_joined if r["split"] == "train"]
    holdout_joined = [r for r in all_joined if r["split"] == "holdout"]

    report = build_report(train_joined, holdout_joined, include_holdout=args.include_holdout)
    print(report)

    # -- Optional JSON output --------------------------------------
    if args.output_json:
        train_acc, *_ = pass_fail_accuracy(train_joined)
        cm = confusion_matrix(train_joined)
        cat_m = per_category_metrics(train_joined)

        payload: Dict = {
            "train": {
                "accuracy": round(train_acc, 4),
                "n": len(train_joined),
                "confusion_matrix": cm,
                "per_category": {
                    cat: {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
                    for cat, m in cat_m.items()
                },
            }
        }
        if holdout_joined and args.include_holdout:
            h_acc, *_ = pass_fail_accuracy(holdout_joined)
            h_cm = confusion_matrix(holdout_joined)
            h_cat = per_category_metrics(holdout_joined)
            payload["holdout"] = {
                "accuracy": round(h_acc, 4),
                "n": len(holdout_joined),
                "confusion_matrix": h_cm,
                "per_category": {
                    cat: {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
                    for cat, m in h_cat.items()
                },
            }

        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Metrics JSON written to: %s", args.output_json)


if __name__ == "__main__":
    main()
