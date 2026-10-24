# AI PDF Assistant

> **A Retrieval-Augmented Generation (RAG) application for asking natural-language questions over PDF documents.**

**Status:** Phase 2 complete — PDF ingest, chunking, embeddings, and FAISS search are working via the API. LLM answer generation (`/ask`) and Streamlit are **not** implemented yet (Phase 3+).

AI PDF Assistant converts PDF documents into a searchable knowledge base: extract text, embed chunks, index them in FAISS, and retrieve relevant context. Grounded LLM answers arrive in Phase 3.

---

## Current development status

| Area | Status |
|------|--------|
| Project layout (`app/`, config, logging) | Done (Phase 0–1) |
| FastAPI `GET /health` + Swagger `/docs` | Done |
| CORS for local Streamlit | Done |
| Config via `.env` / pydantic-settings | Done |
| PDF upload + PyMuPDF extraction | Done (Phase 2) |
| Chunking, embeddings, FAISS persistence | Done (Phase 2) |
| `GET /documents`, `POST /search` | Done (Phase 2) |
| Automated tests (parser, chunks, FAISS, API, E2E) | Done (Phase 2) |
| RAG `/ask` + OpenAI | Planned (Phase 3) |
| Streamlit UI | Planned (Phase 4) |
| Docker / deploy | Planned (Phase 6) |

---

## Planned / target (later phases)

- OpenAI-backed grounded answers (`POST /ask`)
- Streamlit frontend
- Docker / production deploy

---

## Technology stack

| Category | Technologies |
|----------|--------------|
| **Backend** | Python 3.11+, FastAPI, Uvicorn |
| **Config** | pydantic-settings |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2` by default) |
| **Vector DB** | FAISS (local, persisted) |
| **Document processing** | PyMuPDF, LangChain Text Splitters |
| **LLM** (planned) | OpenAI API |
| **Frontend** (planned) | Streamlit |

---

## Project layout

```text
app/
  main.py                 # FastAPI app, CORS, error handlers
  config.py               # Settings from environment
  api/routes.py           # /health, /documents/upload, /documents, /search
  models/schemas.py
  services/
    pdf_parser.py         # Validate + PyMuPDF extract
    chunker.py            # Recursive chunking + metadata
    embeddings.py         # Sentence Transformers (singleton)
    vector_store.py       # FAISS + on-disk metadata
    ingest.py             # Upload → index orchestration
    rag.py                # Stub (Phase 3)
frontend/                 # Streamlit UI (Phase 4)
data/                     # Runtime uploads / indexes (gitignored)
tests/
  fixtures/sample.pdf     # Tiny text PDF for E2E retrieval
```

---

## Local setup

Requires **Python 3.11+** (developed/tested on 3.12).

```bash
git clone https://github.com/SnigdhaJanmanchi/RAG-Based-PDF-Question-Answering-System.git
cd RAG-Based-PDF-Question-Answering-System

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env        # optional; defaults work for Phase 2 (no OpenAI key needed)

uvicorn app.main:app --reload --port 8000
```

Then:

- Health: http://localhost:8000/health
- Swagger: http://localhost:8000/docs

First embedding/search request downloads the Sentence Transformers model (one-time).

### Run tests

```bash
pytest                 # all tests (embedding/E2E marked slow; first run may download the model)
pytest -m "not slow"   # skip model-backed tests
```

---

## Configuration

See `.env.example`. Important Phase 2 variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Recursive splitter |
| `TOP_K` | `5` | Default search result count |
| `DATA_DIR` | `data` | Uploads + FAISS + metadata |
| `MAX_UPLOAD_BYTES` | `10485760` (10 MiB) | Upload size limit |
| `CORS_ORIGINS` | Streamlit localhost | Browser CORS |
| `OPENAI_API_KEY` | empty | Unused until Phase 3 |

Never commit `.env`, uploads, or FAISS indexes (`data/` is gitignored).

---

## API (Phase 2)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness |
| `POST` | `/documents/upload` | Upload PDF → parse → chunk → embed → FAISS |
| `POST` | `/upload` | Alias for `/documents/upload` |
| `GET` | `/documents` | List indexed documents |
| `POST` | `/search` | Semantic search; returns chunk text + metadata (no LLM) |

### Upload response

```json
{
  "document_id": "<uuid>",
  "filename": "report.pdf",
  "page_count": 2,
  "chunk_count": 5,
  "duplicate": false
}
```

Identical file content (SHA-256) returns the existing document with `"duplicate": true` and does not double-index.

### Search request

```json
{ "query": "Where does the indigo phoenix nest?", "top_k": 3, "document_id": null }
```

---

## Limitations (current)

- **No OCR** — image-only / scanned PDFs return a clear 422; text-based PDFs only
- **No LLM answers** — use `/search` for retrieved chunks; `/ask` is Phase 3
- **Encrypted PDFs** are rejected
- **Single-node local FAISS** — not a multi-user managed vector DB
- Failed page extraction aborts the upload (pages are never silently skipped)
