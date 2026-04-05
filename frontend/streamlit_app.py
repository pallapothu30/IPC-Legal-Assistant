from __future__ import annotations

import re

import requests
import streamlit as st

from backend.core.config import settings

API_BASE_URL = f"http://{settings.api_host}:{settings.api_port}"


def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "api_config" not in st.session_state:
        st.session_state.api_config = None


def emphasize_query_terms(text: str, query: str) -> str:
    terms = [term for term in re.findall(r"\w+", query) if len(term) > 3]
    highlighted = text
    for term in sorted(set(terms), key=len, reverse=True):
        highlighted = re.sub(rf"(?i)\b({re.escape(term)})\b", r"**\1**", highlighted)
    return highlighted


def render_sources(source_documents: list[dict], query: str) -> None:
    st.subheader("Source Excerpts")
    for index, doc in enumerate(source_documents, start=1):
        with st.expander(
            f"{index}. Section {doc.get('section_number', 'Unknown')} - {doc.get('section_title', 'Untitled')}",
            expanded=index == 1,
        ):
            st.caption(f"Source: {doc.get('source', 'Unknown source')}")
            st.markdown(emphasize_query_terms(doc.get("content", "")[:600], query))


def request_json(method: str, path: str, payload: dict | None = None) -> dict | None:
    try:
        response = requests.request(
            method=method,
            url=f"{API_BASE_URL}{path}",
            json=payload,
            timeout=settings.api_timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Backend is not running. Start it with: `python scripts/run_backend.py`")
    except requests.exceptions.Timeout:
        st.error("Backend request timed out.")
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc))
        st.error(f"Backend error: {detail}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
    return None


def ensure_backend() -> None:
    health = request_json("GET", "/health")
    if not health:
        st.stop()
    if not health.get("vectorstore_initialized", False):
        st.error("Chroma database not initialized. Run `python scripts/ingest_ipc.py`.")
        st.stop()


def main() -> None:
    st.set_page_config(page_title="IPC Legal Assistant", page_icon="⚖️", layout="wide")
    st.title("IPC Legal Assistant")
    st.caption("Simple frontend for asking IPC questions grounded in retrieved context.")

    init_session_state()
    ensure_backend()

    if st.session_state.api_config is None:
        st.session_state.api_config = request_json("GET", "/config")

    with st.sidebar:
        st.header("Configuration")
        config = st.session_state.api_config or {}
        st.write(f"Model: `{config.get('model', 'N/A')}`")
        st.write(f"Embedding: `{config.get('embedding', 'N/A')}`")
        st.write(f"Retriever top_k: `{config.get('top_k', 'N/A')}`")
        st.write(f"Chroma DB: `{config.get('chroma_db', 'N/A')}`")

    query = st.text_input("Enter your IPC-related question")
    if query:
        with st.spinner("Retrieving relevant IPC sections and generating answer..."):
            result = request_json("POST", "/query", {"question": query, "chat_history": st.session_state.history})

        if result:
            st.session_state.history.append({"role": "user", "content": query})

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Answer")
                st.markdown(result.get("answer", ""))
            with col2:
                st.subheader("Relevant Section(s)")
                for section in result.get("relevant_sections", []) or ["Not identified"]:
                    st.write(section)

            if result.get("source_documents"):
                render_sources(result["source_documents"], query)

            st.session_state.history.append({"role": "assistant", "content": result.get("answer", "")})

    if st.session_state.history:
        st.divider()
        st.subheader("Recent Conversation")
        for item in reversed(st.session_state.history[-6:]):
            label = "Q" if item["role"] == "user" else "A"
            st.markdown(f"**{label}:** {item['content']}")


if __name__ == "__main__":
    main()

