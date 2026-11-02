"""
workflow.py â€“ LangGraph pipeline construction for the Compliance QA Pipeline.

Compiles a StateGraph that threads VideoAudioState through four sequential
nodes, with a conditional routing step after the audit phase.

Graph topology:
    START
      â”‚
      â–¼
  index_video
      â”‚
      â–¼
  retrieve_compliance_rules
      â”‚
      â–¼
  audit_compliance
      â”‚
      â–¼
  generate_report
      â”‚
      â–¼
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


def build_compliance_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph compliance evaluation pipeline.

    Returns the compiled graph ready for ``graph.invoke(initial_state)``.
    """
    builder = StateGraph(VideoAudioState)

    # â”€â”€ Register nodes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    builder.add_node("index_video", index_video)
    builder.add_node("retrieve_compliance_rules", retrieve_compliance_rules)
    builder.add_node("audit_compliance", audit_compliance)
    builder.add_node("generate_report", generate_report)

    # â”€â”€ Wire edges (sequential pipeline) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    builder.add_edge(START, "index_video")
    builder.add_edge("index_video", "retrieve_compliance_rules")
    builder.add_edge("retrieve_compliance_rules", "audit_compliance")
    builder.add_edge("audit_compliance", "generate_report")
    builder.add_edge("generate_report", END)

    compiled = builder.compile()
    logger.info("Compliance graph compiled successfully.")
    return compiled


# Singleton compiled graph â€” import this in server.py and main.py
compliance_graph = build_compliance_graph()
