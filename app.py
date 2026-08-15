"""
app.py
------
Streamlit dashboard for the Finance RAG application.
Designed with a clean, branded AI interface.

Run with: streamlit run app.py
"""

import logging
import time
from pathlib import Path

import streamlit as st

from config import PDF_PATH, validate_config
from ingestion import ingest_pdf, load_existing_vectorstore, IngestionStats
from rag_pipeline import build_rag_chain, query_rag_chain
from kpi_extractor import extract_kpis

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Finance RAG Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Clean, Modern AI Design Tokens & Styling
# --------------------------------------------------------------------------
GEMINI_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-main: #0E1117;
    --bg-surface: #1E2028;
    --bg-surface-variant: #282A36;
    --text-primary: #E3E2E6;
    --text-secondary: #9B9EAB;
    --border-subtle: #2E323D;
    --accent-blue: #A8C7FA;
    --font-stack: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-stack);
    background-color: var(--bg-main);
    color: var(--text-primary);
}

#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background-color: var(--bg-main);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #13151C;
    border-right: 1px solid var(--border-subtle);
}

section[data-testid="stSidebar"] h3 {
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--text-primary);
}

/* Masthead Header */
.ai-header {
    margin-bottom: 32px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-subtle);
}
.ai-header .tag {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.ai-header h1 {
    font-size: 1.75rem;
    font-weight: 500;
    color: var(--text-primary);
    margin: 4px 0 0 0;
    letter-spacing: -0.01em;
}

/* Section Labels */
.section-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-secondary);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 24px 0 12px 0;
}

/* KPI Cards */
.kpi-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px 20px;
    height: 100%;
}
.kpi-card .kpi-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 6px;
}
.kpi-card .kpi-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* Input Fields & Form */
div[data-testid="stTextInput"] input {
    background-color: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    color: var(--text-primary);
    border-radius: 24px;
    padding: 14px 20px;
    font-size: 0.95rem;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent-blue);
    box-shadow: none;
}

div[data-testid="stForm"] {
    border: none;
    padding: 0;
}

.stButton > button {
    background-color: var(--bg-surface-variant);
    color: var(--text-primary);
    font-weight: 500;
    font-size: 0.9rem;
    border-radius: 20px;
    border: 1px solid var(--border-subtle);
    padding: 10px 24px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background-color: var(--accent-blue);
    color: #0E1117;
    border-color: var(--accent-blue);
}

/* AI Answer Panel */
.answer-box {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 20px 24px;
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text-primary);
}

/* Citation / Source Cards */
.source-box {
    background: var(--bg-surface-variant);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 8px;
    font-size: 0.85rem;
    line-height: 1.5;
    color: var(--text-secondary);
}
.source-box .source-meta {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--accent-blue);
    margin-bottom: 4px;
    display: block;
}
</style>
"""
st.markdown(GEMINI_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Session State Initialization
# --------------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "vectorstore": None,
        "rag_chain": None,
        "ingestion_stats": None,
        "kpi_results": None,
        "query_history": [],
        "last_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def try_reconnect_existing_index() -> None:
    if st.session_state.rag_chain is not None:
        return
    try:
        vectorstore = load_existing_vectorstore()
        if vectorstore._collection.count() > 0:  # noqa: SLF001
            st.session_state.vectorstore = vectorstore
            st.session_state.rag_chain = build_rag_chain(vectorstore)
    except Exception:
        pass


try_reconnect_existing_index()


# --------------------------------------------------------------------------
# Sidebar: Controls & Status
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Document Indexing")
    st.caption("Chroma vector store pipeline")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload report PDF", type=["pdf"], label_visibility="collapsed"
    )

    target_path = str(PDF_PATH)
    if uploaded_file is not None:
        Path(PDF_PATH).write_bytes(uploaded_file.getbuffer())
        st.success(f"File loaded: {uploaded_file.name}")

    ingest_clicked = st.button("Run Ingestion", use_container_width=True)

    if ingest_clicked:
        try:
            validate_config()
            with st.spinner("Processing and indexing document..."):
                start = time.time()
                vectorstore, stats = ingest_pdf(target_path)
                chain = build_rag_chain(vectorstore)
                elapsed = time.time() - start

            st.session_state.vectorstore = vectorstore
            st.session_state.rag_chain = chain
            st.session_state.ingestion_stats = stats
            st.session_state.kpi_results = None
            st.success(f"Completed in {elapsed:.1f}s")
        except FileNotFoundError:
            st.error("Document missing. Upload a file or place 'financial_report.pdf' in /data.")
        except EnvironmentError as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001
            st.error(f"Ingestion failed: {e}")

    st.divider()

    stats: IngestionStats | None = st.session_state.ingestion_stats
    if stats:
        st.markdown("**Index Information**")
        st.markdown(
            f"""
            <div style="font-size:0.8rem; color:#9B9EAB; line-height:1.8;">
            Source: <span style="color:#E3E2E6">{stats.source_file}</span><br>
            Pages: <span style="color:#E3E2E6">{stats.total_pages}</span><br>
            Chunks: <span style="color:#E3E2E6">{stats.total_chunks}</span><br>
            Embeddings: <span style="color:#E3E2E6">{stats.embedding_model}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No active document index.")

    st.divider()
    st.markdown("**System Metadata**")
    st.caption("Model: llama-3.1-8b-instant")
    st.caption("Context Top-K: 4")
    st.caption("Chunk Config: 1000 / 200")


# --------------------------------------------------------------------------
# Main Masthead
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="ai-header">
        <div class="tag">Financial Intelligence System</div>
        <h1>Report Analysis & Insights</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Headline Metrics Row
# --------------------------------------------------------------------------
st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)

if st.session_state.rag_chain is None:
    st.markdown(
        '<div class="source-box">Index a document from the sidebar to extract financial metrics.</div>',
        unsafe_allow_html=True,
    )
else:
    if st.session_state.kpi_results is None:
        with st.spinner("Extracting headline metrics..."):
            st.session_state.kpi_results = extract_kpis(st.session_state.rag_chain)

    kpi_cols = st.columns(len(st.session_state.kpi_results))
    for col, kpi in zip(kpi_cols, st.session_state.kpi_results):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{kpi.label}</div>
                    <div class="kpi-value">{kpi.value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------------------
# Query Interface
# --------------------------------------------------------------------------
st.markdown('<div class="section-label">Query Model</div>', unsafe_allow_html=True)

with st.form(key="query_form", clear_on_submit=False):
    query_col, button_col = st.columns([5, 1])
    with query_col:
        user_question = st.text_input(
            "Query Input",
            placeholder="Ask a question about the document...",
            label_visibility="collapsed",
        )
    with button_col:
        submitted = st.form_submit_button("Submit", use_container_width=True)

if submitted:
    if st.session_state.rag_chain is None:
        st.warning("Please index a document prior to submitting queries.")
    elif not user_question.strip():
        st.warning("Enter a prompt to analyze.")
    else:
        with st.spinner("Generating response..."):
            result = query_rag_chain(st.session_state.rag_chain, user_question)
        st.session_state.last_result = {
            "question": user_question,
            "answer": result.get("answer", ""),
            "sources": result.get("context", []),
        }
        st.session_state.query_history.append(user_question)

# --------------------------------------------------------------------------
# Output & Context Citations
# --------------------------------------------------------------------------
if st.session_state.last_result:
    result = st.session_state.last_result

    st.markdown('<div class="section-label">Generated Response</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Retrieved Evidence</div>', unsafe_allow_html=True)

    if not result["sources"]:
        st.markdown(
            '<div class="source-box">No matching context fragments were retrieved.</div>',
            unsafe_allow_html=True,
        )
    else:
        for i, doc in enumerate(result["sources"], start=1):
            page = doc.metadata.get("page", "N/A")
            snippet = doc.page_content.strip().replace("\n", " ")
            st.markdown(
                f"""
                <div class="source-box">
                    <span class="source-meta">Document Snippet {i} • Page {page}</span>
                    {snippet}
                </div>
                """,
                unsafe_allow_html=True,
            )