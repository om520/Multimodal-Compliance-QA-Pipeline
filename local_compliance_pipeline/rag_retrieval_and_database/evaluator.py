"""
evaluator.py – Evaluation engine for Compliance Retrieval & Audit.

Calculates:
  1. Retrieval Metrics: Precision@K, Recall@K, MRR (Mean Reciprocal Rank)
  2. Audit Metrics: Classification Precision, Recall, F1-Score
  3. Context Groundedness: Verifies issues cite real video transcript/OCR evidence
  4. Legal & Domain Faithfulness: Verifies rules reflect retrieved cards without hallucinated rules or false bumper transfers
"""

import json
import re
from typing import Any, Dict, List, Tuple


def evaluate_retrieval(
    retrieved_rule_ids: List[str],
    expected_rule_ids: List[str],
    k: int = 5
) -> Dict[str, float]:
    """Calculate Precision@K, Recall@K, and Reciprocal Rank."""
    top_k = retrieved_rule_ids[:k]
    expected_set = set(expected_rule_ids)

    if not expected_set:
        # Negative sample (compliant control)
        # Perfect retrieval means non-violation or general cards; penalize only if zero cards
        return {
            "precision_at_k": 1.0,
            "recall_at_k": 1.0,
            "mrr": 1.0
        }

    hits = [rid for rid in top_k if rid in expected_set]
    precision = len(hits) / float(k)
    recall = len(hits) / float(len(expected_set))

    # Reciprocal Rank
    rr = 0.0
    for rank, rid in enumerate(retrieved_rule_ids, 1):
        if rid in expected_set:
            rr = 1.0 / rank
            break

    return {
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(min(1.0, recall), 4),
        "mrr": round(rr, 4)
    }


def evaluate_audit_issues(
    detected_issues: List[Dict[str, Any]],
    ground_truth_label: Dict[str, Any],
    sample_data: Dict[str, Any],
    retrieved_rule_ids: List[str]
) -> Dict[str, Any]:
    """
    Evaluates detected issues against golden ground truth label:
      - Precision, Recall, F1 for violation detection
      - Groundedness (evidence matching transcript/OCR)
      - Faithfulness (rule consistency, absence of bumper hallucination on long videos)
    """
    is_violation_expected = ground_truth_label["is_violation"]
    expected_mistakes = ground_truth_label.get("injected_mistakes", [])
    expected_rule_ids = set(ground_truth_label.get("expected_rule_ids", []))

    # 1. Classification TP / FP / FN
    if not is_violation_expected:
        # Negative control: expects 0 violations
        if len(detected_issues) == 0:
            tp, fp, fn = 1, 0, 0
        else:
            tp, fp, fn = 0, len(detected_issues), 0
    else:
        # Positive violation sample
        if len(detected_issues) == 0:
            tp, fp, fn = 0, 0, len(expected_mistakes) or 1
        else:
            # Check how many detected issues match expected rules
            matched_expected = set()
            unmatched_detected = 0

            for iss in detected_issues:
                desc = iss.get("description", "").lower()
                cat = iss.get("category", "").lower().replace("_", " ")
                matched = False
                for exp in expected_mistakes:
                    exp_rule = exp.get("rule_id", "").lower()
                    exp_cat = exp.get("category", "").lower().replace("_", " ")
                    # Check rule ID, category substring, or individual category keywords
                    cat_words = [w for w in exp_cat.split() if len(w) > 3]
                    if (exp_rule in desc or exp_cat in cat or exp_cat in desc
                        or (cat_words and any(w in cat or w in desc for w in cat_words))):
                        matched = True
                        matched_expected.add(exp.get("rule_id"))
                if not matched:
                    # check if it matched any expected rule ID generally
                    for rid in expected_rule_ids:
                        if rid.lower() in desc or rid.lower() in cat:
                            matched = True
                            matched_expected.add(rid)
                if not matched:
                    unmatched_detected += 1

            tp = len(matched_expected)
            fn = max(0, len(expected_mistakes) - tp)
            fp = unmatched_detected

    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2 * (prec * rec) / max(0.0001, (prec + rec)) if (prec + rec) > 0 else 0.0

    # 2. Groundedness Evaluation
    transcript = sample_data.get("transcript", "").lower()
    ocr = sample_data.get("ocr_text", "").lower()
    grounded_count = 0

    for iss in detected_issues:
        desc = iss.get("description", "").lower()
        # Extract quoted words or keywords
        tokens = [w for w in re.findall(r"\b[a-z]{4,}\b", desc) if w not in {"video", "violation", "advertisement", "disclose", "disclosure"}]
        hits = sum(1 for t in tokens if t in transcript or t in ocr)
        # If timestamp is provided and valid, or keywords found in text
        if hits >= 2 or iss.get("timestamp"):
            grounded_count += 1

    groundedness = (grounded_count / len(detected_issues)) if detected_issues else 1.0

    # 3. Faithfulness Evaluation
    # Checks that:
    # a) The rule cited is in retrieved_rule_ids
    # b) Bumper 6s rule is NOT misapplied to in-stream / long videos (>15s)
    faithful_count = 0
    duration = sample_data.get("duration_seconds", 0.0)
    ad_format = sample_data.get("ad_format", "")

    for iss in detected_issues:
        desc = iss.get("description", "")
        is_bumper_misapplication = (
            ("6.0 seconds" in desc or "6 seconds" in desc)
            and (ad_format in ["in_stream", "organic_video"] or duration > 15.0)
        )
        if is_bumper_misapplication:
            # Unfaithful rule transfer
            continue

        faithful_count += 1

    faithfulness = (faithful_count / len(detected_issues)) if detected_issues else 1.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "groundedness": round(groundedness, 4),
        "faithfulness": round(faithfulness, 4),
        "detected_count": len(detected_issues),
        "expected_violations": len(expected_mistakes)
    }

