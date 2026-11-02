"""
eval/ — Compliance QA Pipeline Evaluation Framework
====================================================

This directory contains the complete evaluation framework for measuring the
accuracy of the Compliance QA Pipeline. Everything here is designed to produce
a DEFENSIBLE, HONEST metric that can be reported in a CV, README, or
technical interview.

FILE OVERVIEW
─────────────
ground_truth.csv          Ground truth labels for 32 synthetic clips.
                          Labels were committed BEFORE any pipeline output
                          was seen (this is the key to eval honesty).
                          24 train / 8 holdout.

synthetic_transcripts.py  Synthetic transcript + OCR data for each clip.
                          Injected directly into the pipeline state,
                          bypassing Azure Video Indexer (no real videos needed).

run_eval.py               Batch runner. Invokes the pipeline for each clip
                          and writes predictions.csv.

metrics.py                Reads predictions.csv + ground_truth.csv and
                          computes accuracy, precision, recall, F1 per
                          violation category, confusion matrix, and a
                          defensible result string.

test_eval_harness.py      19 unit tests for the eval harness itself
                          (no Azure creds required).

conftest.py               pytest sys.path bootstrap.

mock_predictions.csv      Output of --mock run (included for reference).
mock_metrics.json         Metrics from mock run (example structure).


STEP-BY-STEP USAGE
──────────────────

Step 1 — Verify harness (no Azure needed):
    python -m pytest eval/test_eval_harness.py -v

Step 2 — Dry-run with mock LLM (no Azure needed):
    python eval/run_eval.py --split train --mock
    python eval/metrics.py --predictions eval/mock_predictions.csv

Step 3 — Real eval on train set (requires Azure OpenAI + Azure Search):
    python eval/run_eval.py --split train --output eval/predictions.csv
    python eval/metrics.py --predictions eval/predictions.csv

Step 4 — Tune the pipeline. Repeat Step 3. NEVER touch holdout.

Step 5 — Final eval on holdout (ONCE only, after finalising the pipeline):
    python eval/run_eval.py --split holdout --output eval/holdout_predictions.csv
    python eval/metrics.py --predictions eval/holdout_predictions.csv --include-holdout

Step 6 — Report the metric:
    Copy the "DEFENSIBLE RESULT STRING" from the metrics output.
    Example: "87% pass/fail accuracy on 8 held-out, self-labeled clips
    (32 total spanning Ad Specs, Brand Safety, Claim Substantiation, FTC
    Disclosure) using ground-truth labels committed before any pipeline
    output was seen."


CLIP CATEGORIES & DISTRIBUTION
───────────────────────────────
Each of the 4 compliance rule categories has 8 clips in the train set:
  - FTC Disclosure          (6 FAIL, 2 PASS)
  - Claim Substantiation    (4 FAIL, 4 PASS)  -- includes subjective edge cases
  - Brand Safety            (4 FAIL, 4 PASS)  -- includes borderline cases
  - Ad Specs                (4 FAIL, 4 PASS)

Holdout (2 per category, mix of PASS/FAIL):
  - FTC Disclosure: clip_026 (PASS), clip_031 (FAIL)
  - Claim Substantiation: clip_025 (FAIL), clip_028 (PASS)
  - Brand Safety: clip_027 (FAIL), clip_030 (PASS)
  - Ad Specs: clip_029 (FAIL), clip_032 (PASS)


IMPORTANT RULES
───────────────
1. NEVER run --split holdout until you are done tuning the pipeline.
2. NEVER adjust the pipeline after seeing holdout predictions.
3. NEVER change ground_truth.csv labels after seeing pipeline output.
   (Labels were committed first — that's what makes the metric honest.)
4. If you retune the pipeline after Step 5, you MUST get fresh labels
   and a new holdout set before reporting a new number.
"""
