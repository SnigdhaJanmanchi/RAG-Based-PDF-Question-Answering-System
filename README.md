# AI PDF Assistant

> **A Retrieval-Augmented Generation (RAG) application for asking natural-language questions over PDF documents.**

**Status:** scaffolding (Phase 0–1). The FastAPI shell, configuration, and project layout are in place. PDF ingest, embeddings, FAISS, RAG answers, and Streamlit are **planned** — not implemented yet.

AI PDF Assistant will convert PDF documents into a searchable knowledge base: extract text, embed chunks, index them in FAISS, retrieve relevant context, and generate grounded answers with an LLM. The backend is modular so parsers, embedding models, vector stores, and LLM providers can be swapped later.

---

## Current development status

| Area | Status |
|------|--------|
| Project layout (`app/`, config, logging) | Done (Phase 0–1) |
| FastAPI `GET /health` + Swagger `/docs` | Done |
| CORS for local Streamlit | Done |
| Config via `.env` / pydantic-settings | Done |
| Automated health tests | Done |
| PDF upload / PyMuPDF extraction | Planned (Phase 2) |
| Chunking, embeddings, FAISS | Planned (Phase 2) |
| RAG `/ask` + OpenAI | Planned (Phase 3) |
| Streamlit UI | Planned (Phase 4) |
| Docker / deploy | Planned (Phase 6) |

---

## Planned / target implementation

- PDF upload and validation
- Text extraction using PyMuPDF
- Recursive text chunking
- Semantic embedding generation with Sentence Transformers
- Vector indexing using FAISS
- Semantic similarity search and context retrieval
- FastAPI REST API with interactive Swagger documentation
- Modular architecture for parsing, embeddings, retrieval, vector storage, and API routing
- OpenAI-backed grounded answers
- Streamlit frontend

---

## Core concepts

- Retrieval-Augmented Generation (RAG)
- Semantic search and vector embeddings
- Vector databases (FAISS)
- Large language models (LLMs)
- Context-aware question answering
- REST API development
- Modular backend architecture

---

## Technology stack

| Category | Technologies |
|----------|--------------|
| **Backend** | Python 3.11+, FastAPI, Uvicorn |
| **Config** | pydantic-settings |
| **AI / ML** (planned) | Sentence Transformers, OpenAI API |
| **Vector DB** (planned) | FAISS |
| **Document processing** (planned) | PyMuPDF, LangChain Text Splitters |
| **Frontend** (planned) | Streamlit |

---

## Project layout

```text
app/
  main.py              # FastAPI app, CORS, error handlers
  config.py            # Settings from environment
  logging_config.py
  api/routes.py        # GET /health (more routes in later phases)
  models/schemas.py
  services/            # Stubs for PDF, chunking, embeddings, FAISS, RAG
frontend/              # Streamlit UI (Phase 4)
data/                  # Runtime uploads / indexes (gitignored)
tests/
```

---

## Local setup (Phase 0–1)

Requires **Python 3.11+**.

```bash
git clone https://github.com/SnigdhaJanmanchi/RAG-Based-PDF-Question-Answering-System.git
cd RAG-Based-PDF-Question-Answering-System

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env        # optional for Phase 0–1; set OPENAI_API_KEY before Phase 3

uvicorn app.main:app --reload --port 8000
```

Then:

- Health: [http://localhost:8000/health](http://localhost:8000/health)
- Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)

### Run tests

```bash
pytest
```

### Notes

- Heavy ML dependencies in `requirements.txt` are pinned for later phases; install may download large packages (Sentence Transformers / FAISS). That is expected.
- PDF upload, indexing, and Q&A endpoints arrive in Phase 2–3. Streamlit arrives in Phase 4.
