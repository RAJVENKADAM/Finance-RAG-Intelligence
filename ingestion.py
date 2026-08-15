"""
ingestion.py
------------
PDF ingestion pipeline: load -> split -> embed -> persist.

Responsible for turning `financial_report.pdf` into a persistent
Chroma vector store using local HuggingFace embeddings.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
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
    """
    Load a PDF file into LangChain Document objects (one per page).

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        A list of Document objects with page-level content and metadata.

    Raises:
        FileNotFoundError: If the PDF does not exist at the given path.
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    logger.info("Loading PDF from %s", pdf_path)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    logger.info("Loaded %d pages", len(pages))
    return pages


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split raw page-level documents into overlapping chunks suitable
    for embedding and retrieval.

    Args:
        documents: Page-level Document objects from the PDF loader.

    Returns:
        A list of chunked Document objects, preserving source metadata
        (e.g. page number) for citation in the UI.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d pages into %d chunks", len(documents), len(chunks))
    return chunks


def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Instantiate the local, free sentence-transformer embedding model.
    Runs on CPU by default so no external API key is required.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """
    Embed chunks and persist them into a local Chroma vector store.
    Clears out any prior collection first so re-ingestion doesn't
    duplicate chunks.

    Args:
        chunks: Chunked Document objects to embed and store.

    Returns:
        A persistent Chroma vector store instance.
    """
    embeddings = get_embedding_function()

    # Safely clear any previous collection instance on disk
    try:
        temp_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        temp_store.delete_collection()
    except Exception:
        pass

    # Build fresh vectorstore from documents
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    logger.info("Persisted %d chunks to Chroma at %s", len(chunks), CHROMA_PERSIST_DIR)
    return vectorstore


def load_existing_vectorstore() -> Chroma:
    """
    Reconnect to an already-persisted Chroma collection without
    re-embedding the source PDF. Used on app restarts.
    """
    embeddings = get_embedding_function()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def ingest_pdf(pdf_path: str) -> Tuple[Chroma, IngestionStats]:
    """
    End-to-end ingestion orchestration: load -> split -> embed -> persist.

    Args:
        pdf_path: Path to the source financial report PDF.

    Returns:
        Tuple of (vectorstore, IngestionStats) for use by the UI layer.
    """
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
