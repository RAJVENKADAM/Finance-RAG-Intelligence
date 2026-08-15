"""
ingestion.py
------------
PDF ingestion pipeline: load -> split -> embed -> persist.

Responsible for turning `financial_report.pdf` into a persistent
Chroma vector store using Serverless Cloud Inference Embeddings.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceEmbeddings
from langchain_chroma import Chroma

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    GROQ_API_KEY,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Lightweight summary returned after ingestion, used by the UI sidebar."""
    source_file: str
    total_pages: int
    total_chunks: int
    embedding_model: str
    persist_dir: str


def load_pdf(pdf_path: str) -> List[Document]:
    """Load a PDF file into LangChain Document objects (one per page)."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    logger.info("Loading PDF from %s", pdf_path)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    logger.info("Loaded %d pages", len(pages))
    return pages


def split_documents(documents: List[Document]) -> List[Document]:
    """Split raw page-level documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d pages into %d chunks", len(documents), len(chunks))
    return chunks


def get_embedding_function() -> HuggingFaceInferenceEmbeddings:
    """Instantiate the Serverless HuggingFace Cloud Inference model."""
    return HuggingFaceInferenceEmbeddings(
        api_key=GROQ_API_KEY,
        model_name=EMBEDDING_MODEL_NAME,
    )


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """Embed chunks and persist them into a local Chroma vector store."""
    embeddings = get_embedding_function()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    
    try:
        vectorstore.delete_collection()
    except Exception:
        pass

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    logger.info("Persisted %d chunks to Chroma at %s", len(chunks), CHROMA_PERSIST_DIR)
    return vectorstore


def load_existing_vectorstore() -> Chroma:
    """Reconnect to an already-persisted Chroma collection."""
    embeddings = get_embedding_function()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def ingest_pdf(pdf_path: str) -> tuple[Chroma, IngestionStats]:
    """End-to-end ingestion orchestration loop."""
    pages = load_pdf(pdf_path)
    chunks = split_documents(pages)
    vectorstore = build_vectorstore(chunks)

    stats = IngestionStats(
        source_file=Path(pdf_path).name,
        total_pages=len(pages),
        total_chunks=len(chunks),
        embedding_model=EMBEDDING_MODEL_NAME,
        persist_dir=CHROMA_PERSIST_DIR,
    )
    return vectorstore, stats
