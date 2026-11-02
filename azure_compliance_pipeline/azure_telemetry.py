"""
telemetry.py â€“ Azure Application Insights + OpenTelemetry setup.

Call ``setup_telemetry()`` once at application startup before any FastAPI
routes are registered. This instruments all HTTP requests, outbound calls,
and LangChain/LangGraph traces automatically.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("brand-guardian")


def setup_telemetry() -> None:
    """
    Configure Azure Monitor OpenTelemetry exporter.

    Requires ``APPLICATIONINSIGHTS_CONNECTION_STRING`` in the environment.
    Silently skips setup if the variable is missing (e.g. local dev without AI).
    """
    conn_str = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "").strip()
    if not conn_str:
        logger.warning("[Telemetry] APPLICATIONINSIGHTS_CONNECTION_STRING not set â€” skipping Azure Monitor setup.")
        return

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor  # type: ignore

        configure_azure_monitor(connection_string=conn_str)
        logger.info("[Telemetry] Azure Application Insights configured successfully.")
    except ImportError:
        logger.warning("[Telemetry] azure-monitor-opentelemetry not installed â€” telemetry disabled.")
    except Exception as exc:
        logger.error("[Telemetry] Failed to configure Azure Monitor: %s", exc, exc_info=True)
