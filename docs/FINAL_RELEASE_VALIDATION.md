# Final Release Validation — AI PDF Assistant

**Date:** 2026-10-26  
**Branch:** `cursor/phase-0-1-scaffold-c77d` (draft PR #1 — **not merged**, **not marked ready for review**)  
**Scope:** CI green, API image size optimization, Compose re-validation, honesty on live OpenAI / production claims.

---

## 1. Verdict

| Gate | Result |
|------|--------|
| Automated tests (`pytest -m "not live"`) | **77 passed**, 1 deselected (local) |
| CI Tests (Python 3.12) after gitignore fix | **SUCCESS** (run after push `12f8efe`) |
| CI Docker image builds | **SUCCESS** on commit `20bcc03` (CPU-torch Dockerfile; all 2 checks green) |
| API image size | **10.2 GB → 2.64 GB** (CPU-only PyTorch; same MiniLM) |
| Embedding model / dims unchanged | **`all-MiniLM-L6-v2`, dim 384** verified in image |
| Compose health / upload / search / restart persistence | **Verified** |
| Live OpenAI `/ask` | **Not verified** — no usable `OPENAI_API_KEY` in environment |
| Auth / public production readiness | **Not claimed** — **no authentication** |
| Cloud deploy / staging URL | **Not performed** (by design) |
| Merge / ready-for-review | **Not done** — awaiting explicit approval |

**Release posture:** Local demo / portfolio packaging is ready. Do **not** treat this as production or public hosting for sensitive PDFs.

---

## 2. CI failure root cause and fix

**Failed job:** `CI / Tests (Python 3.12)` on run [40211996354](https://github.com/SnigdhaJanmanchi/RAG-Based-PDF-Question-Answering-System/actions/runs/40211996354/job/121225000852). Docker job was **skipped** (`needs: [test]`).

**Cause:** `tests/test_e2e_integration.py::test_gitignore_excludes_secrets_and_runtime_data` read `Path("/workspace/.gitignore")`, which exists only in this Cloud Agent VM — not on GitHub Actions runners → `FileNotFoundError`.

**Fix:** Resolve repo root from `Path(__file__).resolve().parents[1]` and read `.gitignore` there. Committed as `12f8efe`.

---

## 3. API image optimization

| Metric | Before | After |
|--------|--------|-------|
| `docker images` size | **10.2 GB** | **2.64 GB** |
| Torch build | `2.14.1+cu130` (CUDA wheels via default PyPI) | `2.14.1+cpu` (`download.pytorch.org/whl/cpu`) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | **unchanged** |
| Embedding dim | 384 | **384** (checked in container) |
| Offline bake | `/app/model_cache` + `HF_HUB_OFFLINE=1` | unchanged; `--network none` encode OK |

**Change:** `Dockerfile.api` installs CPU `torch` from the PyTorch CPU index **before** `pip install -r requirements-api.txt`, so `sentence-transformers` does not pull CUDA wheels. Comment added in `requirements-api.txt`. UI image ≈ **833 MB** (unchanged intent).

---

## 4. Validation matrix (this run)

### Ran successfully

| Check | Result |
|-------|--------|
| `pytest -m "not live"` | **77 passed**, 1 deselected (~10 s) |
| `test_gitignore_excludes_secrets_and_runtime_data` | **Passed** |
| `docker build -f Dockerfile.api` (CPU torch) | **Succeeded** → `ai-pdf-assistant-api:local` **2.64 GB** |
| Offline MiniLM in image (`--network none`) | **Passed** (dim 384) |
| `docker compose up -d --build` | API + UI **healthy** |
| `GET /health` | **200** `status=ok` |
| Streamlit `/_stcore/health` | **200** |
| `POST /documents/upload` (sample.pdf) | **200** (content-hash duplicate OK) |
| `POST /search` query `phoenix` | Top hit page **1**, phoenix text, score ~0.41 |
| `docker compose restart api` + list/search | **Persisted** same `document_id` on `app_data` |
| `POST /ask` without key | Grounded refusal / insufficient path (no paid call) |

### Not run / unverified

| Check | Reason |
|-------|--------|
| Live OpenAI grounded answers | Empty / placeholder `OPENAI_API_KEY` |
| Interactive browser UI walkthrough | Not performed this pass |
| Cloud staging URL | Out of scope — no auto-deploy |
| Public production hardening (auth, TLS, multi-tenant) | Explicitly out of scope |

---

## 5. Security and honesty reminders

- Secrets stay in env / `.env` (gitignored). Never bake keys into images.
- **No auth** on API or UI — unsuitable for public sensitive documents.
- Citations show retrieved context; they do not prove answer correctness.
- CI builds images with `push: false` and **does not deploy**.

---

## 6. How to reproduce locally

```bash
pytest -m "not live"
docker compose build
docker compose up -d
curl -fsS http://localhost:8000/health
# optional: set a real OPENAI_API_KEY in .env, recreate api, then POST /ask
```

---

## 7. Remaining follow-ups (for humans)

1. Optionally run `pytest -m live` and a Compose `/ask` with a real key (do not commit the key).
2. Explicit approval required before marking the draft PR ready or merging.
3. Any public host needs auth, TLS, and a threat model — not provided here.
