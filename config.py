"""
config.py
---------
Centralized configuration for the Finance RAG application.
All tunables live here so the rest of the codebase never hardcodes
paths, model names, or chunking parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a local .env file (if present).
load_dotenv()

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "financial_report.pdf"
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_store")
COLLECTION_NAME = "financial_report"

DATA_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Ingestion / Chunking
# --------------------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --------------------------------------------------------------------------
# Embeddings (local, free, no API key required)
# --------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --------------------------------------------------------------------------
# LLM (Groq)
# --------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
RETRIEVER_TOP_K = 4

# --------------------------------------------------------------------------
# KPI extraction (financial overview cards on the dashboard)
# --------------------------------------------------------------------------
KPI_QUERIES = {
    "Total Revenue": "What is the total revenue reported in this document? Answer with the exact figure and period.",
    "Net Income": "What is the net income (or net profit/loss) reported in this document? Answer with the exact figure and period.",
    "Total Assets": "What are the total assets reported in this document? Answer with the exact figure and period.",
    "Total Liabilities": "What are the total liabilities reported in this document? Answer with the exact figure and period.",
    "EPS": "What is the earnings per share (EPS) reported in this document? Answer with the exact figure and period.",
}


def validate_config() -> None:
    """Raise a clear error early if required secrets are missing."""
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Export it or add it to a .env file "
            "before starting the application."
        )
