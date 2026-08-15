"""
app.py
------
Premium Streamlit dashboard for the Finance RAG application.
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
    page_title="Ledger | Financial Report Intelligence",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark corporate palette styling architecture layout frames
CUSTOM_CSS = """
<style>
@import url('https://googleapis.com');

:root {
    --bg-primary: #0B0E14;
    --bg-panel: #12161F;
    --bg-panel-raised: #171C27;
    --border-hairline: #232935;
    --text-primary: #E8EAED;
    --text-secondary: #8B93A3;
    --accent-brass: #C9A15A;
    --accent-mint: #4FD1AE;
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 0%, rgba(201, 161, 90, 0.05), transparent 40%), var(--bg-primary);
}

.ledger-masthead {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid var(--border-hairline);
    padding-bottom: 14px;
    margin-bottom: 28px;
}
.ledger-masthead h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.9rem;
    color: var(--text-primary);
    margin: 0;
}
.ledger-masthead .eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    color: var(--accent-brass);
}
.ledger-masthead .status-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent-mint);
    border: 1px solid var(--border-hairline);
    padding: 4px 12px;
    border-radius: 999px;
}

.kpi-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-hairline);
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 15px;
}
.kpi-card .kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 8px;
}
.kpi-card .kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    color: var(--text-primary);
}

.section-heading {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    color: var(--accent-brass);
    margin: 30px 0 10px 0;
}

.answer-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-hairline);
    border-left: 3px solid var(--accent-mint);
    border-radius: 10px;
    padding: 22px;
    color: var(--text-primary);
}

.source-card {
    background: var(--bg-panel-raised);
    border: 1px solid var(--border-hairline);
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 10px;
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    color: var(--text-primary);
}
.source-card .source-tag {
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent-brass);
    font-size: 0.7rem;
    margin-bottom: 6px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session_state() -> None:
    defaults = {
        "vectorstore": None,
        "rag_chain": None,
        "ingestion_stats": None,
        "kpi_results": None,
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
        st.session_state.vectorstore = vectorstore
        st.session_state.rag_chain = build_rag_chain(vectorstore)
    except Exception:
        pass


try_reconnect_existing_index()

# --------------------------------------------------------------------------
# Sidebar UI Components
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📑 Document Intelligence")
    st.caption("Cloud-optimized vector ingestion engine")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload financial report (PDF)", type=["pdf"], label_visibility="collapsed"
    )

    target_path = str(PDF_PATH)
    if uploaded_file is not None:
        Path(PDF_PATH).write_bytes(uploaded_file.getbuffer())
        st.success(f"Received: {uploaded_file.name}")

    ingest_clicked = st.button("⚙️  Run Ingestion Pipeline", use_container_width=True)

    if ingest_clicked:
        try:
            validate_config()
            with st.spinner("Loading PDF → Chunking → Cloud Vector Indexing..."):
                start = time.time()
                vectorstore, stats = ingest_pdf(target_path)
                chain = build_rag_chain(vectorstore)
                elapsed = time.time() - start

            st.session_state.vectorstore = vectorstore
            st.session_state.rag_chain = chain
            st.session_state.ingestion_stats = stats
            st.session_state.kpi_results = None  
            st.success(f"Indexed in {elapsed:.1f}s")
        except FileNotFoundError:
            st.error("No file found. Upload a PDF first.")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")

    st.divider()

    stats: IngestionStats = st.session_state.ingestion_stats
    if stats:
        st.markdown("**📂 Index Metadata**")
        st.caption(f"Chunks created: `{stats.total_chunks}`")
        st.caption(f"Source file: `{stats.source_file}`")
    else:
        st.caption("No vector index active.")

# --------------------------------------------------------------------------
# Main Panel View
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="ledger-masthead">
        <div>
            <div class="eyebrow">Enterprise Risk RAG Engine</div>
            <h1>Ledger</h1>
        </div>
        <div class="status-pill">● SECURE CLOUD ACTIVE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Extract metric highlights early if vector store is ready
if st.session_state.rag_chain and not st.session_state.kpi_results:
    with st.spinner("Extracting headline KPIs..."):
        st.session_state.kpi_results = extract_kpis(st.session_state.rag_chain)

# Render Metric Grid
if st.session_state.kpi_results:
    cols = st.columns(len(st.session_state.kpi_results))
    for col, (label, val) in zip(cols, st.session_state.kpi_results.items()):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{val}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown('<div class="section-heading">Financial Question Routing Panel</div>', unsafe_allow_html=True)

with st.form(key="query_form", clear_on_submit=False):
    query_text = st.text_input("Enter analyst prompt", placeholder="What are the structural risks outlined in Section 4?")
    submit_query = st.form_submit_button(label="Execute Query")

if submit_query and query_text:
    if not st.session_state.rag_chain:
        st.error("Please run the ingestion pipeline or upload a file first.")
    else:
        with st.spinner("Running vector similarity retrieval..."):
            try:
                res = query_rag_chain(st.session_state.rag_chain, query_text)
                st.session_state.last_result = {
                    "answer": res.get("answer", "No answer compiled."),
                    "sources": res.get("context", [])
                }
            except Exception as e:
                st.error(f"Execution failed: {str(e)}")

# Display Grounded Answer Screen
if st.session_state.last_result:
    result_data = st.session_state.last_result
    st.markdown("### Grounded Analysis Output")
    st.markdown(f'<div class="answer-card">{result_data["answer"]}</div>', unsafe_allow_html=True)
    
    if result_data["sources"]:
        st.markdown('<div class="section-heading">Retrieved Source Footnotes</div>', unsafe_allow_html=True)
        for idx, doc in enumerate(result_data["sources"]):
            page_num = doc.metadata.get("page", "N/A")
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-tag">Source Chunk [{idx + 1}] — Page {page_num}</div>
                    <div>{doc.page_content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
