# AI PDF Assistant

> **A Retrieval-Augmented Generation (RAG) application for asking natural-language questions over PDF documents.**

**Status:** Phase 6 complete — local Docker Compose packaging + CI. Release validation: CPU-torch API image (~2.64 GB) + CI fix. **No auth; not for public sensitive docs. Not auto-deployed.** See [`docs/FINAL_RELEASE_VALIDATION.md`](docs/FINAL_RELEASE_VALIDATION.md), [`docs/PHASE6_DEPLOYMENT_REPORT.md`](docs/PHASE6_DEPLOYMENT_REPORT.md), and [`docs/PHASE5_VALIDATION_REPORT.md`](docs/PHASE5_VALIDATION_REPORT.md).

AI PDF Assistant converts PDFs into a searchable knowledge base: extract text, embed chunks, index them in FAISS, retrieve relevant context, and generate answers grounded only in that context. Use Swagger or the Streamlit chat UI.

---

## Current development status

| Area | Status |
|------|--------|
| Project layout / config / health | Done (Phase 0–1) |
| PDF upload, chunking, embeddings, FAISS | Done (Phase 2) |
| `GET /documents`, `POST /search` | Done (Phase 2) |
| `POST /ask` grounded OpenAI answers | Done (Phase 3) |
| Streamlit UI (upload, select, chat + citations) | Done (Phase 4) |
| Hardening, retrieval eval, validation report | Done (Phase 5) |
| Docker Compose + GitHub Actions CI | Done (Phase 6) |

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
| **Frontend** | Streamlit |

---

## Project layout

```text
app/                      # FastAPI backend
frontend/                 # Streamlit UI
Dockerfile.api / Dockerfile.ui
docker-compose.yml
.github/workflows/ci.yml  # tests + image builds (no deploy)
data/                     # Runtime uploads / indexes (gitignored)
docs/PHASE5_VALIDATION_REPORT.md
docs/PHASE6_DEPLOYMENT_REPORT.md
docs/FINAL_RELEASE_VALIDATION.md
tests/
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
# Set OPENAI_API_KEY in .env for POST /ask
```

### Terminal A — FastAPI

```bash
uvicorn app.main:app --reload --port 8000
```

- Health: http://localhost:8000/health  
- Swagger: http://localhost:8000/docs  

### Terminal B — Streamlit

```bash
export API_BASE_URL=http://localhost:8000   # optional; this is the default
streamlit run frontend/streamlit_app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Upload PDFs in the sidebar, pick a document or “all”, then chat.

First embedding/upload downloads the Sentence Transformers model (one-time).

### Run tests

```bash
pytest -m "not live"      # full automated suite (OpenAI mocked; no paid API required)
pytest -m live            # optional live OpenAI E2E (only if OPENAI_API_KEY is set)
python scripts/eval_retrieval.py --k 3   # offline Hit@K / Recall@K on fixture PDFs
```

First embedding/model download pulls Sentence Transformers + PyTorch wheels (hundreds of MB). Subsequent runs use the local cache.

Phase 5 validation details: [`docs/PHASE5_VALIDATION_REPORT.md`](docs/PHASE5_VALIDATION_REPORT.md).

---

## Docker Compose (recommended local demo)

**Warning:** There is **no authentication**. Do not put sensitive PDFs on a public host.

```bash
cp .env.example .env
# Set OPENAI_API_KEY for /ask (optional for upload/search only)

docker compose build
docker compose up -d
```

- API health: http://localhost:8000/health  
- Swagger: http://localhost:8000/docs  
- Streamlit: http://localhost:8501  

Uploads + FAISS indexes persist in the Docker volume `app_data`. The API image bakes the MiniLM embedding model and installs **CPU-only** PyTorch (API image ≈ **2.6 GB**; first compose build still downloads wheels + model once).

```bash
docker compose logs -f api
docker compose down          # keeps volume
```

CI (`.github/workflows/ci.yml`) runs `pytest -m "not live"` and builds both images on PRs/`main` — **it does not deploy**.

Portfolio hosting options, resource notes, and troubleshooting: [`docs/PHASE6_DEPLOYMENT_REPORT.md`](docs/PHASE6_DEPLOYMENT_REPORT.md).

---

## Configuration

See `.env.example`. Never commit `.env`, uploads, or FAISS indexes.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | empty | Required on the **API** for `POST /ask` |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `OPENAI_TIMEOUT_SECONDS` | `30` | OpenAI timeout |
| `OPENAI_MAX_RETRIES` | `2` | OpenAI retries |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embeddings |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Recursive splitter |
| `TOP_K` | `5` | Default retrieval count |
| `SIMILARITY_THRESHOLD` | `0.25` | Min cosine score to keep a chunk |
| `CONTEXT_TOKEN_BUDGET` | `2500` | Approx context budget |
| `DATA_DIR` | `data` | Uploads + FAISS + metadata |
| `MAX_UPLOAD_BYTES` | `10485760` | Upload size limit |
| `API_BASE_URL` | `http://localhost:8000` | Streamlit → FastAPI (Compose: `http://api:8000`) |
| `CORS_ORIGINS` | Streamlit localhost | Browser→API CORS (Swagger); tighten for any public UI URL |
| `HF_HUB_OFFLINE` | `1` (Compose) | Use baked embedding weights; set `0` only to refresh from Hugging Face |

The Streamlit app never reads or displays `OPENAI_API_KEY` — only the API server does.

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

There is **no** delete-document HTTP route yet; the Streamlit UI does not show a delete control.

### Sample `POST /ask`

```json
{
  "question": "Where does the indigo phoenix nest?",
  "document_id": null,
  "top_k": 5
}
```

Response includes `answer`, `status` (`ok` | `insufficient_evidence`), `sources` (filename, page, excerpt, score), and `documents`.

---

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| Streamlit: “Cannot reach the API” | Is `uvicorn` running? Is `API_BASE_URL` correct? |
| Upload fails / OCR message | Text-based PDFs only; scanned PDFs need OCR (not supported) |
| Ask returns 503 about API key | Set `OPENAI_API_KEY` in the **API** process `.env`, then restart uvicorn |
| Ask rate limit (429) | Wait and retry; check OpenAI account limits |
| Empty answers / insufficient evidence | Try another question, lower `SIMILARITY_THRESHOLD`, or confirm the PDF was indexed (`GET /documents`) |
| Duplicate upload notices | Expected — identical content is detected by hash; Streamlit also skips re-uploads of the same file bytes in-session |

---

## Limitations (honest)

- **Citations do not prove correctness.** Sources show retrieved context; always verify against the PDF pages.
- Prompts reduce but do not eliminate hallucinations.
- **No OCR**, encrypted PDFs rejected, local FAISS only.
- No document delete API / UI yet.
- **No authentication** — do not expose publicly with sensitive documents.
- API Docker image is still multi-GB (~2.6 GB with CPU PyTorch + MiniLM).
- Live OpenAI ask and interactive browser QA may still be unverified — check Phase 5/6 and final validation reports.
- CI builds images but **does not** auto-deploy to any cloud.
