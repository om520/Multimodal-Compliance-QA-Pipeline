"""
lean_dynamic_chunker.py – Lean Dynamic Semantic Chunker for Compliance RAG.

Optimized to:
  1. Keep each rule card concise (~300-450 characters) to prevent prompt bloat.
  2. Ensure all 8 retrieved rules fit cleanly inside the 3,000-character budget in nodes.py.
  3. Attach rich structured metadata (rule_id, category, severity, applies_to, pages, source).
  4. Upsert cleanly into ChromaDB collection 'compliance_rules'.
"""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

load_dotenv()

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lean-chunker")

# Config
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "compliance_rules")


# ==============================================================================
# LEAN SEMANTIC REGULATORY RULE CARDS (~300-450 characters each)
# ==============================================================================

LEAN_RULE_CARDS = [
    # ── DOCUMENT 1: FTC Influencer Guides (1001a-influencer-guide-508_1.pdf) ──
    {
        "id": "FTC-RULE-001",
        "title": "Material Connection Disclosure",
        "category": "FTC_Disclosure",
        "severity": "critical",
        "applies_to": "transcript,ocr_text",
        "pages": "2,3",
        "source": "1001a-influencer-guide-508_1.pdf",
        "summary": "Mandatory disclosure if endorser has financial, free product, discount, employment, or personal ties to the brand.",
        "text": """[RULE: FTC Material Connection Disclosure]
Source: 1001a-influencer-guide-508_1.pdf (Pages 2-3) | Severity: CRITICAL
Mandate: Endorsers must disclose any financial tie, payment, gifted product, discount, or personal/family relationship with the brand.
Compliance: Required even if evaluations are unbiased or free products were unrequested.
Prohibited: Assuming followers already know about brand relationships; failing to disclose foreign posts targeting US consumers."""
    },
    {
        "id": "FTC-RULE-002",
        "title": "Dual Audio & Visual Video Disclosure Sync",
        "category": "Audio_Visual_Sync",
        "severity": "critical",
        "applies_to": "transcript,ocr_text",
        "pages": "4",
        "source": "1001a-influencer-guide-508_1.pdf",
        "summary": "Video endorsements must place disclosures inside the video in BOTH spoken audio and superimposed text.",
        "text": """[RULE: FTC Video Endorsement - Dual Audio/Visual Sync]
Source: 1001a-influencer-guide-508_1.pdf (Page 4) | Severity: CRITICAL
Mandate: Video disclosures must be inside the video asset itself in BOTH spoken audio AND superimposed text overlay.
Reason: Viewers watching without sound need on-screen text; viewers missing fast text overlays need spoken words.
Prohibited: Placing disclosure only in video description box, on channel bio, or behind 'Show More' fold."""
    },
    {
        "id": "FTC-RULE-003",
        "title": "Conspicuous Placement & Timing",
        "category": "FTC_Disclosure",
        "severity": "critical",
        "applies_to": "transcript,ocr_text,duration_seconds",
        "pages": "4",
        "source": "1001a-influencer-guide-508_1.pdf",
        "summary": "Disclosures must appear at the beginning of the endorsement; end-of-video placement is prohibited.",
        "text": """[RULE: FTC Disclosure Timing & Placement]
Source: 1001a-influencer-guide-508_1.pdf (Page 4) | Severity: CRITICAL
Mandate: Place disclosures prominently with the endorsement itself, before consumers make purchasing decisions.
Prohibited Placements:
- Disclosures placed only at the very end of videos or posts.
- Disclosures hidden in profile pages, video descriptions, or behind 'More' links.
- Live stream endorsements without periodic repeated disclosures."""
    },
    {
        "id": "FTC-RULE-004",
        "title": "Permitted vs Prohibited Language",
        "category": "Language_Clarity",
        "severity": "critical",
        "applies_to": "transcript,ocr_text",
        "pages": "4,5",
        "source": "1001a-influencer-guide-508_1.pdf",
        "summary": "Must use unambiguous terms (#ad, #sponsored, advertisement). Shorthand like #sp, #collab, or 'thanks' is prohibited.",
        "text": """[RULE: FTC Approved vs Prohibited Terminology]
Source: 1001a-influencer-guide-508_1.pdf (Pages 4-5) | Severity: CRITICAL
Approved Terms: '#ad', '#sponsored', 'advertisement', 'Paid partnership with [Brand]', '[Brand]Partner'.
Prohibited Terms: '#sp', '#spon', '#collab', lone 'thanks', lone 'ambassador'.
Hashtags: Never bury or obscure the disclosure inside a cluster of unrelated hashtags or links. Must match endorsement language."""
    },
    {
        "id": "FTC-RULE-005",
        "title": "Truthfulness & Health Claim Substantiation",
        "category": "Claim_Substantiation",
        "severity": "critical",
        "applies_to": "transcript,ocr_text",
        "pages": "6",
        "source": "1001a-influencer-guide-508_1.pdf",
        "summary": "Cannot review un-tried products, falsely praise poor products, or make unsubstantiated medical/health claims.",
        "text": """[RULE: FTC Truthfulness & Scientific Claim Substantiation]
Source: 1001a-influencer-guide-508_1.pdf (Page 6) | Severity: CRITICAL
Mandate: Claims must reflect genuine personal experience and competent scientific substantiation.
Prohibited:
1. Reviewing or endorsing products the creator has not personally used.
2. Falsely praising a product the creator found deficient.
3. Making scientific or medical claims that a product cures, treats, or alters health conditions without rigorous proof."""
    },

    # ── DOCUMENT 2: YouTube Ad Specs Guide 2026 (YouTube_Ad_Specs_Guide_2026.pdf) ──
    {
        "id": "YT-SPEC-001",
        "title": "Skippable In-Stream Video Ads Specs",
        "category": "Ad_Spec_Format",
        "severity": "warning",
        "applies_to": "duration_seconds,aspect_ratio",
        "pages": "1,3",
        "source": "YouTube_Ad_Specs_Guide_2026.pdf",
        "summary": "Skippable ads allow skip after 5s. Recommended 15-20s for awareness/action, 2-3 mins for consideration. 16:9/9:16/1:1.",
        "text": """[SPEC: YouTube Skippable In-Stream Video Ads]
Source: YouTube_Ad_Specs_Guide_2026.pdf (Pages 1, 3) | Severity: WARNING
Duration: Any length. Recommended: 15–20s (Awareness/Action), 2–3 mins (Consideration). Skips after 5s.
Aspect Ratio: 16:9 (1920x1080), 9:16 (1080x1920), 1:1 (1080x1080).
Requirements: Max 256 GB; headline <= 15 chars. Branding and core proposition must be delivered within first 5 seconds."""
    },
    {
        "id": "YT-SPEC-002",
        "title": "Non-Skippable In-Stream Ad Limits (7-15s)",
        "category": "Ad_Spec_Duration",
        "severity": "critical",
        "applies_to": "duration_seconds",
        "pages": "1,3",
        "source": "YouTube_Ad_Specs_Guide_2026.pdf",
        "summary": "Non-skippable ads must be 7-15s strictly (up to 30s on CTV only). Videos over 15s face immediate rejection.",
        "text": """[SPEC: YouTube Non-Skippable In-Stream Ads - 15s Limit]
Source: YouTube_Ad_Specs_Guide_2026.pdf (Pages 1, 3) | Severity: CRITICAL
Duration Limit: Strictly 7 to 15 seconds (standard devices); up to 30 seconds for Connected TV (CTV) only.
Rule: Viewers cannot skip. Any standard ad exceeding 15.0 seconds will be rejected by Google Ads.
Technical: 16:9, 9:16, 1:1 resolutions; max 256 GB; headline <= 15 chars."""
    },
    {
        "id": "YT-SPEC-003",
        "title": "Bumper Ads Strict Limit (<= 6 Seconds)",
        "category": "Ad_Spec_Duration",
        "severity": "critical",
        "applies_to": "duration_seconds",
        "pages": "1,3",
        "source": "YouTube_Ad_Specs_Guide_2026.pdf",
        "summary": "Bumper ads have a strict hard limit of 6.0 seconds maximum and cannot be skipped.",
        "text": """[SPEC: YouTube Bumper Ads - 6s Hard Limit]
Source: YouTube_Ad_Specs_Guide_2026.pdf (Pages 1, 3) | Severity: CRITICAL
Duration Limit: Strictly 6.0 seconds maximum. Non-skippable format.
Violation: Any creative asset measuring 6.1 seconds or longer will fail ad approval immediately.
Technical: 16:9 (1080p), 9:16, 1:1; max 1 GB file size; headline <= 15 chars."""
    },
    {
        "id": "YT-SPEC-004",
        "title": "Shorts Video Ads & UI Safe Zones",
        "category": "Ad_Spec_Shorts",
        "severity": "critical",
        "applies_to": "duration_seconds,aspect_ratio,ocr_text",
        "pages": "1,4",
        "source": "YouTube_Ad_Specs_Guide_2026.pdf",
        "summary": "Shorts require vertical 9:16 (1080x1920), 10-60s. Critical text/CTAs must avoid top 15% and bottom 20% safe zones.",
        "text": """[SPEC: YouTube Shorts Ads & Vertical Safe Zones]
Source: YouTube_Ad_Specs_Guide_2026.pdf (Pages 1, 4) | Severity: CRITICAL
Format: Vertical 9:16 (1080x1920). Duration: 10–60s (up to 3 mins allowed; 10–30s optimal).
Safe Zones: Critical branding, disclaimers, and CTAs must avoid:
- Top 15% (covered by search bar & sound controls).
- Bottom 20% (covered by channel title, handle, and ad badge).
- Right 15% edge (covered by Like/Comment icons)."""
    },
    {
        "id": "YT-SPEC-005",
        "title": "In-Feed Video Ads Specs & Metadata",
        "category": "Ad_Spec_Format",
        "severity": "warning",
        "applies_to": "ocr_text,duration_seconds",
        "pages": "1,4",
        "source": "YouTube_Ad_Specs_Guide_2026.pdf",
        "summary": "In-feed ads appear in search and watch feeds. Headline max 40 chars; description up to 2 lines of 35 chars each.",
        "text": """[SPEC: YouTube In-Feed Video Ads]
Source: YouTube_Ad_Specs_Guide_2026.pdf (Pages 1, 4) | Severity: WARNING
Placement: Search results, watch next feed, and app home feed.
Duration: Any duration; 15–20s recommended.
Text Specs: Headline up to 40 characters (2 lines). Description: up to 2 lines, maximum 35 characters per line. Max 256 GB."""
    },
    {
        "id": "YT-SPEC-006",
        "title": "Audio Ads & -14 LUFS Loudness Standards",
        "category": "Ad_Spec_Audio",
        "severity": "warning",
        "applies_to": "duration_seconds,audio_levels",
        "pages": "1,5,7",
        "source": "YouTube_Ad_Specs_Guide_2026.pdf",
        "summary": "Audio ads max 30s. AAC codec, 44.1/48kHz, 128kbps+, targeting -14 LUFS integrated loudness with -1 dBFS true peak.",
        "text": """[SPEC: YouTube Audio Ads & Loudness Standards]
Source: YouTube_Ad_Specs_Guide_2026.pdf (Pages 1, 5, 7) | Severity: WARNING
Duration: Max 30 seconds (<=15s non-skippable; 16-30s skippable). 16:9 companion visual.
Audio Standards: AAC codec, 44.1 kHz or 48 kHz, minimum 128 kbps stereo.
Loudness: Target -14 LUFS (+-1 LUFS) with -1 dBFS True Peak, matching YouTube automated loudness normalization."""
    },
    {
        "id": "YT-SPEC-007",
        "title": "Call-to-Action (CTA) & Text Overlay Guidelines",
        "category": "Ad_Spec_Visual",
        "severity": "warning",
        "applies_to": "ocr_text,visual_tags",
        "pages": "7,8",
        "source": "YouTube_Ad_Specs_Guide_2026.pdf",
        "summary": "CTA button max 10 characters. On-screen text must have high contrast and mobile legibility without blocking UI.",
        "text": """[SPEC: Call-to-Action (CTA) & Text Overlay Guidelines]
Source: YouTube_Ad_Specs_Guide_2026.pdf (Pages 7, 8) | Severity: WARNING
CTA Requirements: Button text maximum 10 characters ('Shop Now', 'Learn More').
Text Overlay Guidelines: High visual contrast against moving video; legible on smartphone screens without pixelation; must not clash with bottom-right skip button or bottom-left ad badge."""
    },
]


def index_lean_rules():
    """Index lean semantic rule cards into ChromaDB."""
    logger.info("=" * 65)
    logger.info("INDEXING LEAN SEMANTIC COMPLIANCE RULES")
    logger.info("=" * 65)
    logger.info("ChromaDB path   : %s", CHROMA_PERSIST_DIR)
    logger.info("Collection      : %s", CHROMA_COLLECTION_NAME)
    logger.info("Embedding model : %s via %s", OLLAMA_EMBEDDING_MODEL, OLLAMA_BASE_URL)
    logger.info("Total lean rule cards: %d", len(LEAN_RULE_CARDS))

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    ef = OllamaEmbeddingFunction(
        url=OLLAMA_BASE_URL,
        model_name=OLLAMA_EMBEDDING_MODEL,
    )

    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        logger.info("Deleted existing collection '%s'", CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [c["id"] for c in LEAN_RULE_CARDS]
    documents = [c["text"] for c in LEAN_RULE_CARDS]
    metadatas = [
        {
            "rule_id": c["id"],
            "title": c["title"],
            "category": c["category"],
            "severity": c["severity"],
            "applies_to": c["applies_to"],
            "pages": c["pages"],
            "source": c["source"],
            "summary": c["summary"],
            "char_count": len(c["text"]),
        }
        for c in LEAN_RULE_CARDS
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    count = collection.count()
    logger.info("✅ Successfully indexed %d lean rule cards into ChromaDB '%s'!", count, CHROMA_COLLECTION_NAME)


if __name__ == "__main__":
    index_lean_rules()

