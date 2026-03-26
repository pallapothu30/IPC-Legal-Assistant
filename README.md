# IPC Legal Assistant

A production-oriented Retrieval-Augmented Generation (RAG) application for answering questions about the Indian Penal Code (IPC) using:

- Groq `llama3-70b-8192`
- HuggingFace embeddings `BAAI/bge-base-en-v1.5`
- ChromaDB with local persistence
- LangChain orchestration
- Streamlit frontend

## Features

- Loads IPC content from PDF or text files
- Cleans and structures IPC sections with metadata
- Chunks text with `chunk_size=700` and `chunk_overlap=120`
- Stores embeddings in persistent ChromaDB
- Retrieves top `k=5` relevant chunks
- Uses a strict prompt to answer only from retrieved IPC context
- Shows answers, relevant sections, and source excerpts
- Includes conversational memory during the app session

## Demo

[Watch the local app demo](https://github.com/user-attachments/assets/7a5c4992-bd56-42f5-a800-e632d999e798)


## Project Structure

```text
IPC_RAG/
├── app/
├── chains/
├── embeddings/
├── ingestion/
├── retriever/
├── vectorstore/
├── data/
├── config.py
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your Groq API key.
4. Put your IPC source file in `data/` as either `.pdf` or `.txt`.

Example `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=llama-3.3-70b-versatile
IPC_DATA_PATH=data/indian_penal_code.pdf
CHROMA_PERSIST_DIR=chroma_db
COLLECTION_NAME=ipc_sections
```

## Build the Vector Database

```bash
python -m ingestion.ingest_ipc --input data/indian_penal_code.pdf
```

## Run the App

```bash
streamlit run app/main.py
```

## Notes

- The app answers strictly from retrieved IPC context.
- If the needed answer is not in the retrieved context, it returns the fallback message.
- Chroma data is persisted locally in `chroma_db/`.
