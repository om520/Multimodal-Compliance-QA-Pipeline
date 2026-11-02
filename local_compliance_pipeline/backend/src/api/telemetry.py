"""
telemetry.py – Lightweight local telemetry stub.

Replaces Azure Application Insights with simple structured logging.
Call ``setup_telemetry()`` once at application startup.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("brand-guardian")


def setup_telemetry() -> None:
    """
    Configure logging level from environment.

    Previously wired to Azure Monitor OpenTelemetry — now uses stdlib logging
    so no cloud credentials are required for local development.
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Telemetry: local logging initialised at level %s", log_level)
