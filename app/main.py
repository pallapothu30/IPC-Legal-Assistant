from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from chains.ipc_chain import build_rag_chain
from config import settings


def init_session_state() -> None:
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = build_rag_chain()
    if "history" not in st.session_state:
        st.session_state.history = []


def emphasize_query_terms(text: str, query: str) -> str:
    terms = [term for term in re.findall(r"\w+", query) if len(term) > 3]
    highlighted = text
    for term in sorted(set(terms), key=len, reverse=True):
        highlighted = re.sub(
            rf"(?i)\b({re.escape(term)})\b",
            r"**\1**",
            highlighted,
        )
    return highlighted


def extract_sections_from_answer(answer: str, fallback_sections: list[str]) -> list[str]:
    matches = re.findall(r"Section(?:s)?\s*[:\-]?\s*([0-9A-Z,\sand]+)", answer, flags=re.IGNORECASE)
    if matches:
        normalized = ", ".join(match.strip() for match in matches if match.strip())
        return [normalized]
    return fallback_sections


def render_sources(source_documents: list, query: str) -> None:
    st.subheader("Source Excerpts")
    for index, doc in enumerate(source_documents, start=1):
        section_number = doc.metadata.get("section_number", "Unknown")
        section_title = doc.metadata.get("section_title", "Untitled")
        source = doc.metadata.get("source", "Unknown source")
        excerpt = emphasize_query_terms(doc.page_content[:600], query)

        with st.expander(f"{index}. Section {section_number} - {section_title}", expanded=index == 1):
            st.caption(f"Source: {source}")
            st.markdown(excerpt)


def ensure_vectorstore_exists() -> None:
    persist_dir = Path(settings.persist_directory)
    if not persist_dir.exists():
        st.error(
            "No Chroma database found. Run `python -m ingestion.ingest_ipc --input <path-to-ipc-file>` first."
        )
        st.stop()


def main() -> None:
    st.set_page_config(page_title="IPC Legal Assistant", page_icon="⚖️", layout="wide")
    st.title("IPC Legal Assistant")
    st.caption("Ask questions grounded strictly in retrieved IPC context.")

    ensure_vectorstore_exists()
    init_session_state()

    with st.sidebar:
        st.header("Configuration")
        st.write(f"Model: `{settings.groq_model_name}`")
        st.write(f"Embedding: `{settings.embedding_model_name}`")
        st.write(f"Retriever top_k: `{settings.top_k}`")
        st.write(f"Chroma DB: `{settings.persist_directory}`")

    query = st.text_input("Enter your IPC-related question")

    if query:
        with st.spinner("Retrieving relevant IPC sections and generating answer..."):
            result = st.session_state.rag_chain.invoke({"question": query})

        answer = result["answer"]
        source_documents = result.get("source_documents", [])
        fallback_sections = []
        for doc in source_documents:
            section_number = doc.metadata.get("section_number")
            if section_number and section_number not in fallback_sections:
                fallback_sections.append(section_number)

        relevant_sections = extract_sections_from_answer(answer, fallback_sections)
        st.session_state.history.append({"query": query, "result": result})

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Answer")
            st.markdown(answer)
        with col2:
            st.subheader("Relevant Section(s)")
            if relevant_sections:
                for section in relevant_sections:
                    st.write(section)
            else:
                st.write("Not identified")

        if source_documents:
            render_sources(source_documents, query)

    if st.session_state.history:
        st.divider()
        st.subheader("Conversation History")
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(f"**Q:** {item['query']}")
            st.markdown(item["result"]["answer"])


if __name__ == "__main__":
    main()

