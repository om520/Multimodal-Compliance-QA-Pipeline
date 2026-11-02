"""
video_indexer.py – Local multimodal media analysis service.

Replaces Azure Video Indexer with fully local tools:
  - yt-dlp          → download YouTube videos
  - faster-whisper  → speech-to-text transcription
  - PaddleOCR       → on-screen text extraction from video frames
  - BLIP            → visual scene / object captioning

No cloud credentials required.
"""

import logging
import os
import subprocess
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("brand-guardian")

# ── Lazy-load heavy models so startup is fast ─────────────────────────────────
_whisper_model = None
_ocr_model = None
_blip_processor = None
_blip_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        model_size = os.environ.get("WHISPER_MODEL", "base")
        logger.info("[Whisper] Loading model: %s", model_size)
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _whisper_model


def _get_ocr():
    global _ocr_model
    if _ocr_model is None:
        import torch  # Pre-load PyTorch DLLs before PaddlePaddle on Windows
        from paddleocr import PaddleOCR
        logger.info("[PaddleOCR] Loading OCR model…")
        _ocr_model = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _ocr_model


def _get_blip():
    global _blip_processor, _blip_model
    if _blip_model is None:
        import torch
        from PIL import Image
        from transformers import BlipForConditionalGeneration, BlipProcessor

        vision_model = os.environ.get(
            "VISION_MODEL", "Salesforce/blip-image-captioning-base"
        )
        logger.info("[BLIP] Loading vision model: %s", vision_model)
        _blip_processor = BlipProcessor.from_pretrained(vision_model)
        _blip_model = BlipForConditionalGeneration.from_pretrained(vision_model)
        _blip_model.eval()
    return _blip_processor, _blip_model


def get_model_load_states() -> Dict[str, Dict[str, str]]:
    """Return non-invasive status for the lazily loaded media models.

    This intentionally does not load or download models. Use it from health
    checks so opening the browser never starts a multi-gigabyte download.
    """
    def state_for(model: object, dependency: str, model_name: str, load_detail: str) -> Dict[str, str]:
        if model is not None:
            return {"state": "loaded", "model": model_name, "detail": "Loaded in this server process."}
        if find_spec(dependency) is None:
            return {
                "state": "missing_dependency",
                "model": model_name,
                "detail": f"Python package '{dependency}' is not installed.",
            }
        return {"state": "available", "model": model_name, "detail": load_detail}

    return {
        "whisper": state_for(
            _whisper_model,
            "faster_whisper",
            os.environ.get("WHISPER_MODEL", "base"),
            "Package is installed; model loads on the first transcription.",
        ),
        "paddleocr": state_for(
            _ocr_model,
            "paddleocr",
            "en",
            "Package is installed; model loads on the first OCR request.",
        ),
        "blip": state_for(
            _blip_model,
            "transformers",
            os.environ.get("VISION_MODEL", "Salesforce/blip-image-captioning-base"),
            "Package is installed; model loads on the first visual-caption request.",
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────


class LocalMediaService:
    """
    Fully local equivalent of Azure Video Indexer.

    Extracts transcript, OCR text, visual captions, and duration
    from a video file using open-source models only.
    """

    def __init__(self) -> None:
        self.frame_sample_seconds: int = int(
            os.environ.get("FRAME_SAMPLE_SECONDS", "10")
        )

    # ── Step 1 – Download ─────────────────────────────────────────────────────

    def download_youtube_video(
        self, url: str, output_path: str = "temp_audit_video.mp4"
    ) -> str:
        """Download a YouTube video to disk using yt-dlp. Returns local path."""
        logger.info("[yt-dlp] Downloading: %s", url)
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", output_path,
            "--no-playlist",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr}")
        logger.info("[yt-dlp] Saved to: %s", output_path)
        return output_path

    # ── Step 2 – Transcription ────────────────────────────────────────────────

    def transcribe(self, video_path: str) -> str:
        """
        Transcribe spoken audio using faster-whisper.
        Returns a single string with timestamps like [00:01:23] sentence…
        """
        logger.info("[Whisper] Transcribing: %s", video_path)
        model = _get_whisper()
        segments, info = model.transcribe(video_path, beam_size=5)

        lines = []
        for seg in segments:
            ts = time.strftime("%H:%M:%S", time.gmtime(seg.start))
            lines.append(f"[{ts}] {seg.text.strip()}")

        transcript = "\n".join(lines)
        logger.info(
            "[Whisper] Done — %d segments, language=%s", len(lines), info.language
        )
        return transcript

    # ── Step 3 – OCR ──────────────────────────────────────────────────────────

    def extract_ocr(self, video_path: str) -> str:
        """
        Sample frames every FRAME_SAMPLE_SECONDS and run PaddleOCR on each.
        Returns deduplicated on-screen text joined by newlines.
        """
        logger.info("[OCR] Sampling frames from: %s", video_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = int(fps * self.frame_sample_seconds)

        ocr = _get_ocr()
        texts: List[str] = []
        seen: set = set()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                result = ocr.ocr(frame, cls=True)
                if result and result[0]:
                    for line in result[0]:
                        if line and len(line) > 1 and line[1] and len(line[1]) > 0:
                            text = str(line[1][0]).strip()
                            if text and text not in seen:
                                seen.add(text)
                                ts = time.strftime(
                                    "%H:%M:%S", time.gmtime(frame_idx / fps)
                                )
                                texts.append(f"[{ts}] {text}")
            frame_idx += 1

        cap.release()
        logger.info("[OCR] Extracted %d unique text lines", len(texts))
        return "\n".join(texts)

    # ── Step 4 – Visual captioning ────────────────────────────────────────────

    def extract_visual_tags(self, video_path: str) -> List[str]:
        """
        Sample frames every FRAME_SAMPLE_SECONDS and generate captions with BLIP.
        Returns a list of unique caption strings.
        """
        import torch
        from PIL import Image

        logger.info("[BLIP] Generating visual captions from: %s", video_path)
        processor, model = _get_blip()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = int(fps * self.frame_sample_seconds)

        captions: List[str] = []
        seen: set = set()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                # Convert BGR→RGB→PIL
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                inputs = processor(pil_img, return_tensors="pt")
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=50)
                caption = processor.decode(out[0], skip_special_tokens=True).strip()
                if caption and caption not in seen:
                    seen.add(caption)
                    ts = time.strftime("%H:%M:%S", time.gmtime(frame_idx / fps))
                    captions.append(f"[{ts}] {caption}")
            frame_idx += 1

        cap.release()
        logger.info("[BLIP] Generated %d unique captions", len(captions))
        return captions

    # ── Step 5 – Duration ─────────────────────────────────────────────────────

    def get_duration(self, video_path: str) -> float:
        """Return the total duration of the video in seconds."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0.0

    # ── Convenience: full pipeline ────────────────────────────────────────────

    def analyse(self, video_path: str) -> Dict[str, Any]:
        """Run the full local analysis pipeline on an already-downloaded file."""
        return {
            "transcript": self.transcribe(video_path),
            "ocr_text": self.extract_ocr(video_path),
            "visual_tags": self.extract_visual_tags(video_path),
            "duration_seconds": self.get_duration(video_path),
        }
