from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from config import settings
from retriever.ipc_retriever import get_ipc_retriever

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


def build_rag_chain(memory: ConversationBufferMemory | None = None) -> ConversationalRetrievalChain:
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your environment or .env file.")

    llm = ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=settings.groq_model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=get_ipc_retriever(),
        memory=memory or get_memory(),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": build_qa_prompt()},
        verbose=False,
    )
