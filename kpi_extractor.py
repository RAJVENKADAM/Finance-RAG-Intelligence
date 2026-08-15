"""
kpi_extractor.py
-----------------
Derives headline financial KPI cards (Revenue, Net Income, Total Assets,
Total Liabilities, EPS) by running targeted queries through the RAG chain
once per ingestion. Results are cached by the UI layer (session_state)
so the LLM is not re-queried on every rerun.
"""

import logging
from dataclasses import dataclass
from typing import List

from langchain_core.runnables import Runnable

from config import KPI_QUERIES
from rag_pipeline import query_rag_chain

logger = logging.getLogger(__name__)


@dataclass
class KPIResult:
    """A single KPI card's data: label, extracted value, and supporting source."""
    label: str
    value: str
    source_snippet: str
    source_page: str


def extract_kpis(chain: Runnable) -> List[KPIResult]:
    """
    Run each configured KPI query through the RAG chain and package
    the results for rendering as dashboard metric cards.

    Args:
        chain: A built retrieval chain (see rag_pipeline.build_rag_chain).

    Returns:
        List of KPIResult, one per configured KPI query.
    """
    results: List[KPIResult] = []

    for label, question in KPI_QUERIES.items():
        try:
            response = query_rag_chain(chain, question)
            answer = response.get("answer", "Not available").strip()
            context_docs = response.get("context", [])

            if context_docs:
                top_doc = context_docs[0]
                snippet = top_doc.page_content[:220].strip() + "..."
                page = str(top_doc.metadata.get("page", "N/A"))
            else:
                snippet, page = "No supporting context found.", "N/A"

            results.append(
                KPIResult(
                    label=label,
                    value=answer,
                    source_snippet=snippet,
                    source_page=page,
                )
            )
        except Exception as exc:  # noqa: BLE001 - surface gracefully in the UI
            logger.exception("Failed to extract KPI '%s'", label)
            results.append(
                KPIResult(
                    label=label,
                    value="Unavailable",
                    source_snippet=f"Extraction error: {exc}",
                    source_page="N/A",
                )
            )

    return results
