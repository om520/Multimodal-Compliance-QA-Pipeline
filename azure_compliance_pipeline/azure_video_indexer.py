"""
video_indexer.py â€“ Azure Video Indexer service client.

Handles the full media lifecycle:
  1. Download YouTube video via yt-dlp
  2. Upload to Azure Blob Storage
  3. Submit to Azure Video Indexer
  4. Poll until indexing completes
  5. Extract structured features (transcript, OCR, visual tags)
"""

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("brand-guardian")


class VideoIndexerService:
    """Wrapper around the Azure Video Indexer ARM-based REST API."""

    VI_API_BASE = "https://api.videoindexer.ai"
    ARM_BASE = "https://management.azure.com"

    def __init__(self) -> None:
        self.account_name: str = os.environ["AZURE_VI_NAME"]
        self.account_id: str = os.environ["AZURE_VI_ACCOUNT_ID"]
        self.location: str = os.environ.get("AZURE_VI_LOCATION", "eastus")
        self.subscription_id: str = os.environ["AZURE_SUBSCRIPTION_ID"]
        self.resource_group: str = os.environ["AZURE_RESOURCE_GROUP"]
        self.storage_conn_str: str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        self._access_token: Optional[str] = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _get_access_token(self) -> str:
        """Obtain a Video Indexer account-level access token via ARM."""
        if self._access_token:
            return self._access_token

        credential = DefaultAzureCredential()
        arm_token = credential.get_token("https://management.azure.com/.default").token

        url = (
            f"{self.ARM_BASE}/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.VideoIndexer/accounts/{self.account_name}"
            f"/generateAccessToken?api-version=2024-01-01"
        )
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {arm_token}", "Content-Type": "application/json"},
            json={"permissionType": "Contributor", "scope": "Account"},
            timeout=30,
        )
        resp.raise_for_status()
        self._access_token = resp.json()["accessToken"]
        return self._access_token  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Step 1 â€“ Download YouTube video
    # ------------------------------------------------------------------

    def download_youtube_video(self, url: str, output_path: str = "temp_video.mp4") -> str:
        """Download a YouTube video using yt-dlp and return the local file path."""
        import yt_dlp  # type: ignore

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        logger.info("[VideoIndexer] Downloaded video to: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Step 2 â€“ Upload to Azure Blob Storage
    # ------------------------------------------------------------------

    def upload_to_blob(self, local_path: str, blob_name: Optional[str] = None) -> str:
        """Upload a local file to Azure Blob Storage and return its URL."""
        container_name = "compliance-videos"
        blob_name = blob_name or f"{uuid.uuid4()}.mp4"

        client = BlobServiceClient.from_connection_string(self.storage_conn_str)

        # Ensure container exists
        container_client = client.get_container_client(container_name)
        try:
            container_client.create_container()
        except Exception:
            pass  # Already exists

        with open(local_path, "rb") as data:
            container_client.upload_blob(
                name=blob_name,
                data=data,
                overwrite=True,
                content_settings=ContentSettings(content_type="video/mp4"),
            )

        blob_url = f"https://{client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
        logger.info("[VideoIndexer] Uploaded blob: %s", blob_url)
        return blob_url

    # ------------------------------------------------------------------
    # Step 3 â€“ Submit to Azure Video Indexer
    # ------------------------------------------------------------------

    def submit_video(self, video_id: str, video_url: str, video_name: str) -> str:
        """Submit a video URL to Azure Video Indexer and return the VI video ID."""
        token = self._get_access_token()
        url = f"{self.VI_API_BASE}/{self.location}/Accounts/{self.account_id}/Videos"

        params = {
            "accessToken": token,
            "name": video_name,
            "videoUrl": video_url,
            "externalId": video_id,
            "indexingPreset": "Default",
            "streamingPreset": "NoStreaming",
        }

        resp = requests.post(url, params=params, timeout=60)
        resp.raise_for_status()
        vi_video_id: str = resp.json()["id"]
        logger.info("[VideoIndexer] Submitted video, VI id: %s", vi_video_id)
        return vi_video_id

    # ------------------------------------------------------------------
    # Step 4 â€“ Poll for indexing completion
    # ------------------------------------------------------------------

    def wait_for_indexing(
        self, vi_video_id: str, poll_interval: int = 15, max_wait: int = 900
    ) -> Dict[str, Any]:
        """Poll Azure Video Indexer until indexing completes; return insights dict."""
        token = self._get_access_token()
        url = f"{self.VI_API_BASE}/{self.location}/Accounts/{self.account_id}/Videos/{vi_video_id}/Index"

        elapsed = 0
        while elapsed < max_wait:
            resp = requests.get(url, params={"accessToken": token}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            state = data.get("state", "")
            logger.info("[VideoIndexer] Indexing state: %s (elapsed %ds)", state, elapsed)

            if state == "Processed":
                return data
            if state == "Failed":
                raise RuntimeError(f"Azure Video Indexer failed for video {vi_video_id}")

            time.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Video indexing timed out after {max_wait}s for {vi_video_id}")

    # ------------------------------------------------------------------
    # Step 5 â€“ Feature extraction helpers
    # ------------------------------------------------------------------

    def extract_transcript(self, insights: Dict[str, Any]) -> str:
        """Concatenate all transcript blocks into a single string."""
        try:
            videos = insights.get("videos", [])
            lines: List[str] = []
            for video in videos:
                for block in video.get("insights", {}).get("transcript", []):
                    text = block.get("text", "").strip()
                    if text:
                        lines.append(text)
            return " ".join(lines)
        except Exception as exc:
            logger.warning("[VideoIndexer] Transcript extraction failed: %s", exc)
            return ""

    def extract_ocr(self, insights: Dict[str, Any]) -> List[str]:
        """Return a list of on-screen text snippets from OCR results."""
        try:
            snippets: List[str] = []
            for video in insights.get("videos", []):
                for block in video.get("insights", {}).get("ocr", []):
                    text = block.get("text", "").strip()
                    ts = block.get("instances", [{}])[0].get("start", "")
                    if text:
                        snippets.append(f"[{ts}] {text}" if ts else text)
            return snippets
        except Exception as exc:
            logger.warning("[VideoIndexer] OCR extraction failed: %s", exc)
            return []

    def extract_visual_tags(self, insights: Dict[str, Any]) -> List[str]:
        """Return a deduplicated list of visual / scene-level labels."""
        try:
            tags: List[str] = []
            for video in insights.get("videos", []):
                for label in video.get("insights", {}).get("labels", []):
                    name = label.get("name", "").strip()
                    if name and name not in tags:
                        tags.append(name)
            return tags
        except Exception as exc:
            logger.warning("[VideoIndexer] Visual tag extraction failed: %s", exc)
            return []

    def extract_duration(self, insights: Dict[str, Any]) -> float:
        """Return video duration in seconds (float)."""
        try:
            duration_str: str = insights.get("durationInSeconds", "0")
            return float(duration_str)
        except Exception:
            return 0.0
