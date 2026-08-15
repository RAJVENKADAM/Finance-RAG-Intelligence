# Finance RAG — Financial Report Intelligence Dashboard

A streamlined Retrieval-Augmented Generation (RAG) system built for financial report processing, KPI auto-extraction, and grounded document Q&A using LangChain, ChromaDB, and Groq's LLM engine.

---

## Project Structure

```text
finance_rag/
├── app.py                # Streamlit dashboard UI (entry point)
├── config.py             # Centralized configuration & environment validation
├── ingestion.py          # PDF parsing, text splitting, embedding & ChromaDB persistence
├── rag_pipeline.py       # LangChain retrieval chain & document Q&A setup
├── kpi_extractor.py      # Automated extraction of headline financial metrics
├── requirements.txt      # Project dependencies
├── .env.example          # Template for environment variables
├── data/
│   └── financial_report.pdf # Default source PDF directory
└── chroma_store/         # Persistent vector store database (auto-created)