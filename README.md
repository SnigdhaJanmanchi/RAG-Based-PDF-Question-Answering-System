# AI PDF Assistant

> **A Retrieval-Augmented Generation (RAG) application for asking natural-language questions over PDF documents.**

**Status:** Phase 3 complete — upload, FAISS search, and grounded OpenAI answers via `POST /ask` work locally. Streamlit UI and Docker deploy are **not** implemented (Phase 4+).

AI PDF Assistant converts PDFs into a searchable knowledge base: extract text, embed chunks, index them in FAISS, retrieve relevant context, and generate answers grounded only in that context.

---

## Current development status

| Area | Status |
|------|--------|
| Project layout / config / health | Done (Phase 0–1) |
| PDF upload, chunking, embeddings, FAISS | Done (Phase 2) |
| `GET /documents`, `POST /search` | Done (Phase 2) |
| `POST /ask` grounded OpenAI answers | Done (Phase 3) |
| Automated tests (mocked OpenAI + Phase 2 suite) | Done (Phase 3) |
| Streamlit UI | Planned (Phase 4) |
| Docker / deploy | Planned (Phase 6) |

---

## Technology stack

| Category | Technologies |
|----------|--------------|
| **Backend** | Python 3.11+, FastAPI, Uvicorn |
| **Config** | pydantic-settings |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2` by default) |
| **Vector DB** | FAISS (local, persisted) |
| **Document processing** | PyMuPDF, LangChain Text Splitters |
| **LLM** | OpenAI API (`gpt-4o-mini` by default) |
| **Frontend** (planned) | Streamlit |

---

## Project layout

```text
app/
  main.py / config.py / api/routes.py
  services/
    pdf_parser.py, chunker.py, embeddings.py, vector_store.py, ingest.py
    openai_client.py   # Official SDK wrapper (timeouts + retries)
    rag.py             # Retrieve → grounded prompt → OpenAI
frontend/              # Streamlit (Phase 4)
data/                  # Runtime uploads / indexes (gitignored)
tests/
  fixtures/sample.pdf
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
cp .env.example .env
# Edit .env and set OPENAI_API_KEY for POST /ask (not required for upload/search)

uvicorn app.main:app --reload --port 8000
```

Then open Swagger at http://localhost:8000/docs — upload a PDF, then call `/ask` or `/search`.

First embedding request downloads the Sentence Transformers model (one-time).

### Run tests

```bash
pytest                    # full suite (OpenAI calls are mocked; no paid API required)
pytest -m "not live"      # same — live tests skip without OPENAI_API_KEY
pytest -m live            # optional live OpenAI E2E (only if OPENAI_API_KEY is set)
```

---

## Configuration

See `.env.example`. Never commit `.env`, uploads, or FAISS indexes.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | empty | Required for `POST /ask` |
| `OPENAI_MODEL` | `gpt-4o-mini` | Cost-effective chat model |
| `OPENAI_TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `OPENAI_MAX_RETRIES` | `2` | SDK retries for transient failures |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embeddings |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Recursive splitter |
| `TOP_K` | `5` | Default retrieval count |
| `SIMILARITY_THRESHOLD` | `0.25` | Min cosine score to keep a chunk |
| `CONTEXT_TOKEN_BUDGET` | `2500` | Approx token budget for context |
| `DATA_DIR` | `data` | Uploads + FAISS + metadata |
| `MAX_UPLOAD_BYTES` | `10485760` | Upload size limit (10 MiB) |

---

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness |
| `POST` | `/documents/upload` | Upload PDF → index |
| `POST` | `/upload` | Alias for upload |
| `GET` | `/documents` | List indexed documents |
| `POST` | `/search` | Semantic search (no LLM) |
| `POST` | `/ask` | Grounded RAG answer (OpenAI) |

### Sample `POST /ask` request

```json
{
  "question": "Where does the indigo phoenix nest?",
  "document_id": null,
  "top_k": 5
}
```

`document_id` is optional — omit it to search all indexed documents.

### Sample `POST /ask` response (`status: ok`)

```json
{
  "question": "Where does the indigo phoenix nest?",
  "answer": "According to the document, the indigo phoenix nests at Lake Azure.",
  "status": "ok",
  "sources": [
    {
      "document_id": "<uuid>",
      "filename": "sample.pdf",
      "page": 1,
      "chunk_id": "<uuid>:p1:c0",
      "excerpt": "The indigo phoenix nests at Lake Azure in the year 1842.",
      "score": 0.81
    }
  ],
  "documents": [
    { "document_id": "<uuid>", "filename": "sample.pdf" }
  ]
}
```

### Insufficient evidence

When no chunks pass the similarity threshold (or the index is empty), the API returns HTTP 200 with:

```json
{
  "status": "insufficient_evidence",
  "answer": "I could not find enough relevant information in the indexed PDF content to answer that question.",
  "sources": [],
  "documents": []
}
```

No OpenAI call is made in that case. Missing `OPENAI_API_KEY` when evidence exists returns **503**.

---

## Limitations (honest)

- **Prompts reduce but do not eliminate hallucinations.** The model is instructed to use only retrieved excerpts; always verify answers against cited pages.
- **No OCR** — scanned/image-only PDFs return 422.
- **Encrypted PDFs** are rejected.
- **PDF text is treated as untrusted data** (prompt-injection resistant design), but a determined adversarial PDF can still confuse the model.
- **Single-node local FAISS** — not multi-tenant or managed vector DB.
- Streamlit UI and Docker packaging are not included yet.
