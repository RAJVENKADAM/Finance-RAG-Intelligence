"""
rag_pipeline.py
----------------
RAG orchestration layer: wires the Chroma retriever, the Groq LLM,
and an enterprise-grade QA prompt together via LangChain's
`create_retrieval_chain` / `create_stuff_documents_chain` (v0.3+ API).
"""

import logging

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from config import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    RETRIEVER_TOP_K,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Enterprise QA prompt
# --------------------------------------------------------------------------
# Grounded, citation-aware, and explicit about refusing to hallucinate
# when the retrieved context does not contain the answer.
QA_SYSTEM_PROMPT = """You are a senior financial analyst assistant embedded in an
enterprise document intelligence platform. Answer the user's question using ONLY
the information provided in the context below, which was retrieved from an
official financial report.

Guidelines:
- Be precise, concise, and quantitative where the context allows it.
- Cite figures exactly as they appear in the context (currency, units, periods).
- If the context does not contain enough information to answer confidently,
  state clearly that the report does not provide that information. Do not
  speculate or fabricate figures.
- Use a professional, neutral tone suitable for an executive audience.

Context:
{context}"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)


def get_llm() -> ChatGroq:
    """
    Instantiate the Groq-hosted chat model.
    The API key is pulled securely from the environment (never hardcoded).
    """
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


def build_rag_chain(vectorstore: Chroma) -> Runnable:
    """
    Assemble the full retrieval-augmented generation chain.

    Args:
        vectorstore: A populated Chroma vector store.

    Returns:
        A runnable chain that accepts {"input": <question>} and returns
        {"answer": ..., "context": [Document, ...]}.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_TOP_K},
    )

    llm = get_llm()

    # Combines retrieved documents into a single prompt ("stuffing" strategy).
    document_chain = create_stuff_documents_chain(llm=llm, prompt=QA_PROMPT)

    # Wires retriever -> document_chain into one retrieval chain.
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain,
    )

    logger.info("RAG chain built with top_k=%d on model=%s", RETRIEVER_TOP_K, GROQ_MODEL_NAME)
    return retrieval_chain


def query_rag_chain(chain: Runnable, question: str) -> dict:
    """
    Execute a single query against the RAG chain.

    Args:
        chain: The runnable built by `build_rag_chain`.
        question: Natural-language question from the user.

    Returns:
        Dict with keys "answer" (str) and "context" (List[Document]).
    """
    result = chain.invoke({"input": question})
    return result
