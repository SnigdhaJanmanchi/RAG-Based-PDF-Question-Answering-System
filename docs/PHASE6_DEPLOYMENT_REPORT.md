# Phase 6 Deployment Report — AI PDF Assistant

**Date:** 2026-10-25  
**Branch:** `cursor/phase-0-1-scaffold-c77d` (draft PR — **not merged**, **not auto-deployed**)  
**Scope:** Docker images, Compose stack, GitHub Actions CI, deploy docs. No cloud publish.

---

## 1. Executive summary

Phase 6 packaging is implemented and validated locally where Docker was available.

| Claim | Status |
|-------|--------|
| Separate API + UI Dockerfiles (non-root, slim base) | Done |
| `docker-compose.yml` with volume + healthchecks | Done |
| GitHub Actions CI (pytest + image builds, **no deploy**) | Done |
| Compose health (API + Streamlit) | **Verified** |
| Upload → search in Compose | **Verified** |
| FAISS persistence after `docker compose restart api` | **Verified** |
| Live OpenAI in containers | **Not verified** (no key) |
| Manual browser walkthrough | **Not run** |
| Cloud / public hosting | **Not performed** (by design) |
| Production readiness / public internet | **Not claimed** — **no auth** |

---

## 2. Implementation

| Artifact | Purpose |
|----------|---------|
| `Dockerfile.api` | FastAPI + embeddings; bakes MiniLM into `/app/model_cache`; runs as uid 10001 |
| `Dockerfile.ui` | Streamlit + httpx only; `API_BASE_URL` points at `http://api:8000` in Compose |
| `requirements-api.txt` / `requirements-ui.txt` | Split deps for smaller UI image |
| `.dockerignore` | Excludes `.env`, venvs, `data/*`, caches, tests |
| `docker-compose.yml` | `api` + `ui`, `app_data` volume, healthchecks, `depends_on` |
| `.github/workflows/ci.yml` | On PR/`main`: pytest (`not live`) + build both images (`push: false`) |

**Security notes**

- API keys via env / `.env` only — never `COPY`’d into images.
- Do **not** expose this stack publicly with sensitive PDFs (**no authentication**).
- CORS defaults to local Streamlit origins; tighten `CORS_ORIGINS` if you put a public UI URL in front of Swagger.
- Compose must **not** mount a volume over `/app/model_cache` (would hide baked weights).

**Resources / model loading**

- API image ≈ **10.2 GB** (PyTorch + CUDA wheel deps + MiniLM). UI ≈ **834 MB**.
- First request loads MiniLM from the image cache (seconds). Soft Compose limits: API 4 GB / 2 CPU; UI 1 GB / 0.5 CPU.
- Set `HF_HUB_OFFLINE=0` only if you intentionally download/refresh models and the host has egress.

---

## 3. Validation results

### Ran successfully

| Check | Result |
|-------|--------|
| `pytest -m "not live"` | **77 passed**, 1 deselected |
| `docker build -f Dockerfile.api` | **Succeeded** (`ai-pdf-assistant-api:local`) |
| `docker build -f Dockerfile.ui` | **Succeeded** (`ai-pdf-assistant-ui:local`) |
| `docker compose up -d` | **Succeeded**; both services **healthy** |
| `GET /health` | **200** `{"status":"ok",...}` |
| Streamlit `/_stcore/health` | **200** |
| `POST /documents/upload` (sample.pdf) | **200** (~3.7 s) |
| `POST /search` | **200** with page-1 phoenix hit |
| Restart API + re-list/search | **Passed** — same `document_id` retained on `app_data` volume |

### Not run / unverified

| Check | Reason |
|-------|--------|
| Live OpenAI `/ask` in Compose | No `OPENAI_API_KEY` |
| Interactive browser UI session | Not performed |
| Cloud deploy (Railway/Render/Fly/etc.) | Explicitly out of scope |
| CI workflow on GitHub Actions runners | Defined in repo; execution depends on GitHub after push |

---

## 4. How to run (Docker Compose)

```bash
cp .env.example .env
# Set OPENAI_API_KEY in .env for /ask (optional for upload/search)

docker compose build
docker compose up -d

# API  http://localhost:8000/health
# UI   http://localhost:8501
# Swagger http://localhost:8000/docs

docker compose logs -f api
docker compose down          # keeps named volume app_data
docker volume rm <project>_app_data   # wipe indexes/uploads if needed
```

**Persistence:** uploads + FAISS live in Docker volume `app_data` → `/app/data`.

---

## 5. Portfolio / hosting options (manual)

Examples only — **do not auto-deploy from this agent**:

| Option | Notes |
|--------|-------|
| Local Compose | Best for demos; free |
| A single VPS (Hetzner/DigitalOcean) | Install Docker; `compose up`; ~4 GB+ RAM for API |
| Railway / Render / Fly.io | Map two services + volume; set secrets in the dashboard |
| Streamlit Community Cloud | UI only; point `API_BASE_URL` at a separately hosted API |

**Cost drivers:** API RAM/disk (10 GB image + model), OpenAI token usage, idle always-on hosts.

---

## 6. Troubleshooting

| Symptom | Fix |
|---------|-----|
| Upload hangs / HF errors | Ensure image was built with baked model; do not wipe `/app/model_cache`; keep `HF_HUB_OFFLINE=1` |
| UI “Cannot reach the API” | Confirm `api` healthy; UI env `API_BASE_URL=http://api:8000` |
| `/ask` → 503 | Set `OPENAI_API_KEY` on the **api** service and recreate |
| Lost documents after recreate | You removed the `app_data` volume |
| OOM kills | Raise `API_MEM_LIMIT` / host RAM |

---

## 7. Remaining limitations

- **No auth / multi-tenant isolation** — unsuitable for public sensitive docs.
- Large API image (torch stack); CPU-only torch index not applied yet.
- Live OpenAI + browser QA still environment-dependent.
- CI does not deploy; staging URL was **not** created by this work.
- Exit criterion “staging URL answers questions” from the original plan is **not met** (no cloud staging) — local Compose validation substitutes for packaging readiness.

---

## 8. Files added/changed (Phase 6)

`Dockerfile.api`, `Dockerfile.ui`, `docker-compose.yml`, `.dockerignore`, `requirements-api.txt`, `requirements-ui.txt`, `.github/workflows/ci.yml`, `.env.example`, `README.md`, plan doc, this report.
