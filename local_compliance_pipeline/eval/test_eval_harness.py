"""
test_eval_harness.py – Unit tests for the evaluation framework itself.

Tests that do NOT require Azure credentials:
  - Ground truth CSV has expected structure
  - Synthetic transcripts cover all 32 clips
  - Metrics functions are correct on known inputs
  - Mock runner produces expected output shape

Run with:
    cd complianceQApipeline
    python -m pytest eval/test_eval_harness.py -v
"""

import csv
import sys
from pathlib import Path

import pytest

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parents[3]  # Desktop/multimodel/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# 1. Ground Truth CSV sanity checks
# ---------------------------------------------------------------------------

class TestGroundTruth:
    GT_PATH = _EVAL_DIR / "ground_truth.csv"

    def _load(self):
        with open(self.GT_PATH, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_file_exists(self):
        assert self.GT_PATH.exists(), "ground_truth.csv not found"

    def test_expected_columns(self):
        rows = self._load()
        required = {"clip_id", "gt_pass_fail", "gt_violation_category", "gt_severity", "split"}
        assert required.issubset(set(rows[0].keys()))

    def test_total_clips(self):
        rows = self._load()
        assert len(rows) == 32, f"Expected 32 clips, got {len(rows)}"

    def test_holdout_count(self):
        rows = self._load()
        holdout = [r for r in rows if r["split"] == "holdout"]
        assert len(holdout) == 8, f"Expected 8 holdout clips, got {len(holdout)}"

    def test_train_count(self):
        rows = self._load()
        train = [r for r in rows if r["split"] == "train"]
        assert len(train) == 24, f"Expected 24 train clips, got {len(train)}"

    def test_valid_verdicts(self):
        rows = self._load()
        for r in rows:
            assert r["gt_pass_fail"] in ("PASS", "FAIL"), (
                f"Unexpected verdict '{r['gt_pass_fail']}' for {r['clip_id']}"
            )

    def test_valid_severities(self):
        rows = self._load()
        for r in rows:
            assert r["gt_severity"] in ("critical", "warning", "info"), (
                f"Unexpected severity '{r['gt_severity']}' for {r['clip_id']}"
            )

    def test_four_categories_covered(self):
        rows = self._load()
        categories = {r["gt_violation_category"] for r in rows}
        expected = {"FTC Disclosure", "Claim Substantiation", "Brand Safety", "Ad Specs"}
        assert expected == categories, f"Categories mismatch: {categories}"

    def test_balanced_pass_fail_in_train(self):
        rows = self._load()
        train = [r for r in rows if r["split"] == "train"]
        pass_count = sum(1 for r in train if r["gt_pass_fail"] == "PASS")
        fail_count = sum(1 for r in train if r["gt_pass_fail"] == "FAIL")
        # Ratio should be roughly 1:2 to 1:3 (more violations than passes)
        assert pass_count > 0, "No PASS clips in train set"
        assert fail_count > 0, "No FAIL clips in train set"
        ratio = fail_count / pass_count
        assert 1.0 <= ratio <= 4.0, (
            f"Pass/fail ratio in train set seems off: {pass_count} PASS, {fail_count} FAIL"
        )


# ---------------------------------------------------------------------------
# 2. Synthetic Transcripts coverage
# ---------------------------------------------------------------------------

class TestSyntheticTranscripts:
    def test_all_clips_have_transcripts(self):
        from complianceQApipeline.eval.synthetic_transcripts import CLIP_TRANSCRIPTS
        gt_path = _EVAL_DIR / "ground_truth.csv"
        with open(gt_path, newline="", encoding="utf-8") as f:
            gt_ids = {row["clip_id"] for row in csv.DictReader(f)}
        missing = gt_ids - set(CLIP_TRANSCRIPTS.keys())
        assert not missing, f"Missing transcripts for: {missing}"

    def test_transcript_fields_present(self):
        from complianceQApipeline.eval.synthetic_transcripts import CLIP_TRANSCRIPTS
        for cid, data in CLIP_TRANSCRIPTS.items():
            assert "transcript" in data, f"{cid} missing 'transcript'"
            assert "ocr_text" in data, f"{cid} missing 'ocr_text'"
            assert "visual_tags" in data, f"{cid} missing 'visual_tags'"
            assert isinstance(data["transcript"], str), f"{cid} transcript must be str"
            assert isinstance(data["ocr_text"], list), f"{cid} ocr_text must be list"


# ---------------------------------------------------------------------------
# 3. Metrics logic unit tests
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for the metrics.py computation functions on known inputs."""

    def _make_rows(self, pairs):
        """pairs = [(gt, pred, category), ...]"""
        return [
            {
                "gt_pass_fail": gt,
                "pred_pass_fail": pred,
                "gt_violation_category": cat,
                "pred_violation_categories": cat if pred == "FAIL" else "",
                "split": "train",
            }
            for gt, pred, cat in pairs
        ]

    def test_perfect_accuracy(self):
        from complianceQApipeline.eval.metrics import pass_fail_accuracy
        rows = self._make_rows([("PASS", "PASS", "FTC Disclosure")] * 5 +
                               [("FAIL", "FAIL", "Ad Specs")] * 5)
        acc, correct, n = pass_fail_accuracy(rows)
        assert acc == 1.0
        assert correct == 10
        assert n == 10

    def test_zero_accuracy(self):
        from complianceQApipeline.eval.metrics import pass_fail_accuracy
        rows = self._make_rows([("PASS", "FAIL", "FTC Disclosure")] * 4 +
                               [("FAIL", "PASS", "Ad Specs")] * 4)
        acc, correct, n = pass_fail_accuracy(rows)
        assert acc == 0.0
        assert correct == 0

    def test_confusion_matrix(self):
        from complianceQApipeline.eval.metrics import confusion_matrix
        rows = self._make_rows([
            ("FAIL", "FAIL", "FTC Disclosure"),   # TP
            ("FAIL", "FAIL", "Ad Specs"),           # TP
            ("PASS", "FAIL", "Brand Safety"),       # FP
            ("FAIL", "PASS", "Claim Substantiation"),  # FN
            ("PASS", "PASS", "FTC Disclosure"),     # TN
        ])
        cm = confusion_matrix(rows)
        assert cm["TP"] == 2
        assert cm["FP"] == 1
        assert cm["FN"] == 1
        assert cm["TN"] == 1

    def test_precision_recall_f1_perfect(self):
        from complianceQApipeline.eval.metrics import precision_recall_f1
        p, r, f = precision_recall_f1(tp=10, fp=0, fn=0)
        assert p == 1.0
        assert r == 1.0
        assert f == 1.0

    def test_precision_recall_f1_all_fn(self):
        from complianceQApipeline.eval.metrics import precision_recall_f1
        p, r, f = precision_recall_f1(tp=0, fp=0, fn=5)
        assert p == 0.0
        assert r == 0.0
        assert f == 0.0

    def test_per_category_returns_all_categories(self):
        from complianceQApipeline.eval.metrics import per_category_metrics
        rows = self._make_rows([
            ("FAIL", "FAIL", "FTC Disclosure"),
            ("FAIL", "FAIL", "Ad Specs"),
            ("FAIL", "PASS", "Brand Safety"),
            ("PASS", "PASS", "Claim Substantiation"),
        ])
        result = per_category_metrics(rows)
        assert set(result.keys()) == {"FTC Disclosure", "Ad Specs", "Brand Safety", "Claim Substantiation"}


# ---------------------------------------------------------------------------
# 4. Mock runner integration test
# ---------------------------------------------------------------------------

class TestMockRunner:
    def test_mock_run_produces_csv(self, tmp_path):
        """Run mock eval on train split and verify predictions CSV is written."""
        from complianceQApipeline.eval.run_eval import run_evaluation

        out_csv = tmp_path / "test_predictions.csv"
        result_path = run_evaluation(
            split="train",
            output_path=out_csv,
            mock=True,
            delay_seconds=0.0,
        )
        assert result_path.exists()

        with open(result_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 24, f"Expected 24 train predictions, got {len(rows)}"

        required_fields = {"clip_id", "pred_pass_fail", "pred_audit_score", "num_issues"}
        for row in rows:
            assert required_fields.issubset(set(row.keys())), f"Missing fields in row: {row}"
            assert row["pred_pass_fail"] in ("PASS", "FAIL"), f"Invalid verdict: {row}"

    def test_mock_accuracy_reasonable(self, tmp_path):
        """Mock is seeded at 85% accuracy — result should be within 65%-100%."""
        from complianceQApipeline.eval.run_eval import run_evaluation
        from complianceQApipeline.eval.metrics import pass_fail_accuracy, join_data, load_csv

        out_csv = tmp_path / "mock_preds.csv"
        run_evaluation(split="train", output_path=out_csv, mock=True, delay_seconds=0.0)

        gt_rows  = load_csv(_EVAL_DIR / "ground_truth.csv")
        pred_rows = load_csv(out_csv)
        joined = join_data(gt_rows, pred_rows)
        train_joined = [r for r in joined if r["split"] == "train"]

        acc, _, _ = pass_fail_accuracy(train_joined)
        assert 0.60 <= acc <= 1.0, f"Mock accuracy out of expected range: {acc:.1%}"
