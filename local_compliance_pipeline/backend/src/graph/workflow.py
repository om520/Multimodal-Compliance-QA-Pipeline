"""
workflow.py – LangGraph pipeline construction for the local Compliance QA Pipeline.

Graph topology:
    START
      │
      ▼
  index_video          (yt-dlp + faster-whisper + PaddleOCR + BLIP)
      │
      ▼
  retrieve_compliance_rules   (ChromaDB + Ollama embeddings)
      │
      ▼
  audit_compliance     (Ollama LLM)
      │
      ▼
  generate_report      (Ollama LLM)
      │
      ▼
    END
"""

import logging

from langgraph.graph import END, START, StateGraph

from backend.src.graph.nodes import (
    audit_compliance,
    generate_report,
    index_video,
    retrieve_compliance_rules,
)
from backend.src.graph.state import VideoAudioState

logger = logging.getLogger("brand-guardian")


def build_graph():
    """Compile and return the LangGraph compliance pipeline."""
    builder = StateGraph(VideoAudioState)

    builder.add_node("index_video", index_video)
    builder.add_node("retrieve_compliance_rules", retrieve_compliance_rules)
    builder.add_node("audit_compliance", audit_compliance)
    builder.add_node("generate_report", generate_report)

    builder.add_edge(START, "index_video")
    builder.add_edge("index_video", "retrieve_compliance_rules")
    builder.add_edge("retrieve_compliance_rules", "audit_compliance")
    builder.add_edge("audit_compliance", "generate_report")
    builder.add_edge("generate_report", END)

    graph = builder.compile()
    logger.info("LangGraph compliance pipeline compiled successfully.")
    return graph


# Module-level compiled graph — imported by server.py and main.py
compliance_graph = build_graph()

# Backwards-compatible name used by the original CLI and integrations.
app = compliance_graph
