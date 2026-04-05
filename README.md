# IPC Legal Assistant

A simple Retrieval-Augmented Generation (RAG) app for answering questions about the Indian Penal Code (IPC) with a Streamlit frontend and FastAPI backend.

## Structure

```text
IPC_RAG/
├── backend/
│   ├── api/
│   │   ├── app.py
│   │   └── routes/
│   ├── core/
│   │   ├── config.py
│   │   └── exceptions.py
│   ├── schemas/
│   │   └── query.py
│   └── services/
│       ├── ingestion_service.py
│       └── rag_service.py
├── frontend/
│   └── streamlit_app.py
├── scripts/
│   ├── ingest_ipc.py
│   ├── run_backend.py
│   └── run_frontend.py
├── data/
├── chroma_db/
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example` and configure:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=llama-3.3-70b-versatile
IPC_DATA_PATH=data/indian_penal_code.pdf
CHROMA_PERSIST_DIR=chroma_db
COLLECTION_NAME=ipc_sections
API_HOST=127.0.0.1
API_PORT=8000
API_TIMEOUT=30
```

## Build the Vector Database

```bash
python scripts/ingest_ipc.py --input data/indian_penal_code.pdf
```

## Run the App

Terminal 1:

```bash
python scripts/run_backend.py
```

Terminal 2:

```bash
python scripts/run_frontend.py
```

Direct commands:

```bash
uvicorn backend.api.app:app --host 127.0.0.1 --port 8000
python -m streamlit run frontend/streamlit_app.py
```

## API Endpoints

- `GET /`
- `GET /health`
- `GET /config`
- `POST /query`

## Notes

- The app answers strictly from retrieved IPC context.
- ChromaDB is stored locally in `chroma_db/`.
- The frontend talks to the backend over HTTP using the configured host and port.
