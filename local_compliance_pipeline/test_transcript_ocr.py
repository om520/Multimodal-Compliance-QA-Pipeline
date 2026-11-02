"""
test_transcript_ocr.py – Test transcript and OCR extraction on compliance videos.

Runs LocalMediaService.transcribe() and extract_ocr() on each video in test_videos/,
saves individual results, then writes a combined output for verification.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test-pipeline")

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from backend.src.services.video_indexer import LocalMediaService


def main():
    test_dir = Path(__file__).parent / "test_videos"
    output_dir = Path(__file__).parent / "test_results"
    output_dir.mkdir(exist_ok=True)

    videos = sorted(test_dir.glob("*.mp4"))
    if not videos:
        logger.error("No .mp4 files found in %s", test_dir)
        sys.exit(1)

    logger.info("Found %d videos to process", len(videos))
    service = LocalMediaService()

    all_results = {}

    for video_path in videos:
        name = video_path.stem
        logger.info("=" * 60)
        logger.info("Processing: %s", name)
        logger.info("=" * 60)

        result = {"video": name, "transcript": "", "ocr_text": "", "errors": []}

        # --- Transcript ---
        try:
            t0 = time.time()
            transcript = service.transcribe(str(video_path))
            elapsed = time.time() - t0
            result["transcript"] = transcript
            result["transcript_time_s"] = round(elapsed, 1)
            logger.info("[%s] Transcript: %d chars in %.1fs", name, len(transcript), elapsed)
        except Exception as e:
            err = f"Transcript error: {e}"
            result["errors"].append(err)
            logger.error("[%s] %s", name, err)

        # --- OCR ---
        try:
            t0 = time.time()
            ocr_text = service.extract_ocr(str(video_path))
            elapsed = time.time() - t0
            result["ocr_text"] = ocr_text
            result["ocr_time_s"] = round(elapsed, 1)
            logger.info("[%s] OCR: %d chars in %.1fs", name, len(ocr_text), elapsed)
        except Exception as e:
            err = f"OCR error: {e}"
            result["errors"].append(err)
            logger.error("[%s] %s", name, err)

        # --- Duration ---
        try:
            result["duration_seconds"] = service.get_duration(str(video_path))
        except Exception as e:
            result["errors"].append(f"Duration error: {e}")

        all_results[name] = result

        # Save individual result
        out_file = output_dir / f"{name}_result.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("[%s] Saved to %s", name, out_file)

    # Save combined results
    combined_file = output_dir / "all_results.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 70)
    print("  TRANSCRIPT & OCR EXTRACTION RESULTS")
    print("=" * 70)
    for name, res in all_results.items():
        print(f"\n--- {name} ---")
        print(f"  Duration: {res.get('duration_seconds', 0):.1f}s")
        print(f"  Transcript: {len(res['transcript'])} chars ({res.get('transcript_time_s', '?')}s)")
        print(f"  OCR text:   {len(res['ocr_text'])} chars ({res.get('ocr_time_s', '?')}s)")
        if res["errors"]:
            print(f"  ⚠️ Errors: {res['errors']}")
        # Print first 500 chars of transcript
        if res["transcript"]:
            print(f"  Transcript preview:\n    {res['transcript'][:500]}")
        if res["ocr_text"]:
            print(f"  OCR preview:\n    {res['ocr_text'][:300]}")
    print("=" * 70)
    print(f"\nFull results saved to: {output_dir}")


if __name__ == "__main__":
    main()

