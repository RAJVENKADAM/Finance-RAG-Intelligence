"""
kpi_extractor.py
-----------------
Derives headline financial KPI cards by running targeted queries through the RAG chain.
"""

import logging
from dataclasses import dataclass
from typing import Dict

from langchain_core.runnables import Runnable
from config import KPI_QUERIES
from rag_pipeline import query_rag_chain

logger = logging.getLogger(__name__)


def extract_kpis(chain: Runnable) -> Dict[str, str]:
    """Run configured KPI queries and return clean label-value strings for the cards."""
    results: Dict[str, str] = {}

    for label, question in KPI_QUERIES.items():
        try:
            response = query_rag_chain(chain, question)
            answer = response.get("answer", "Not available").strip()
            results[label] = answer
        except Exception as exc:
            logger.exception("Failed to extract KPI '%s'", label)
            results[label] = "Unavailable"

    return results
