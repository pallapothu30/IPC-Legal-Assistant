from __future__ import annotations

import re

from fastapi import HTTPException
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

from backend.core.config import settings
from backend.core.exceptions import RagConfigurationError, VectorstoreNotInitializedError
from backend.schemas.query import ConfigResponse, QueryRequest, QueryResponse, SourceDocument
from backend.services.ingestion_service import get_vectorstore

SYSTEM_PROMPT = """You are a legal assistant specialized in the Indian Penal Code (IPC).

Your job is to answer user questions ONLY using the provided context.

Rules:

1. Use ONLY the given context. Do not use prior knowledge.
2. If the answer is not present in the context, say: 'The provided IPC context is insufficient to answer this question.'
3. Always mention relevant IPC section numbers.
4. Be precise, formal, and legally accurate.
5. Do NOT hallucinate or make up laws.
6. Prefer concise explanations over long paragraphs.

Output format:

* Answer
* Relevant Section(s)
* Explanation (based strictly on context)"""

_rag_chain: ConversationalRetrievalChain | None = None


def build_qa_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "question"],
        template=(
            f"{SYSTEM_PROMPT}\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Response:"
        ),
    )


def get_memory() -> ConversationBufferMemory:
    return ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )


def get_rag_chain() -> ConversationalRetrievalChain:
    global _rag_chain

    if _rag_chain is not None:
        return _rag_chain

    if not settings.groq_api_key:
        raise RagConfigurationError("GROQ_API_KEY is missing. Add it to your environment or .env file.")

    llm = ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=settings.groq_model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    _rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=get_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": settings.top_k}),
        memory=get_memory(),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": build_qa_prompt()},
        verbose=False,
    )
    return _rag_chain


def extract_sections_from_answer(answer: str, fallback_sections: list[str]) -> list[str]:
    matches = re.findall(r"Section(?:s)?\s*[:\-]?\s*([0-9A-Z,\sand]+)", answer, flags=re.IGNORECASE)
    if matches:
        return [", ".join(match.strip() for match in matches if match.strip())]
    return fallback_sections


def get_app_config() -> ConfigResponse:
    return ConfigResponse(
        model=settings.groq_model_name,
        embedding=settings.embedding_model_name,
        top_k=settings.top_k,
        chroma_db=str(settings.persist_directory),
    )


def get_answer(request: QueryRequest) -> QueryResponse:
    if not settings.persist_directory.exists():
        raise HTTPException(status_code=503, detail="Chroma database not initialized. Run ingestion first.")

    try:
        result = get_rag_chain().invoke({"question": request.question})
        answer = result.get("answer", "")
        source_documents_raw = result.get("source_documents", [])

        fallback_sections: list[str] = []
        for doc in source_documents_raw:
            section_number = doc.metadata.get("section_number")
            if section_number and section_number not in fallback_sections:
                fallback_sections.append(section_number)

        source_documents = [
            SourceDocument(
                content=doc.page_content,
                section_number=doc.metadata.get("section_number"),
                section_title=doc.metadata.get("section_title"),
                source=doc.metadata.get("source", "Unknown source"),
            )
            for doc in source_documents_raw
        ]

        return QueryResponse(
            answer=answer,
            source_documents=source_documents,
            relevant_sections=extract_sections_from_answer(answer, fallback_sections),
        )
    except HTTPException:
        raise
    except (RagConfigurationError, VectorstoreNotInitializedError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing query: {exc}") from exc

