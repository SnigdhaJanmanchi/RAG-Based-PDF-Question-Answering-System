# Phase 5 Validation Report — AI PDF Assistant

**Date:** 2026-10-25  
**Branch:** `cursor/phase-0-1-scaffold-c77d`  
**Python:** 3.12.3  
**Scope:** Hardening, retrieval quality, expanded fixtures/tests, security review. **No Docker/deploy (Phase 6).**

---

## 1. Executive summary

Phase 5 validation **passed automated checks** for ingest → FAISS retrieval → (mocked) ask, Streamlit API contracts, persistence after store reload, duplicate uploads, and edge-case PDFs.

| Claim | Status |
|-------|--------|
| Local RAG upload/search/ask path | Verified (ask with **mocked** OpenAI) |
| Streamlit ↔ FastAPI contracts | Verified via automated integration + AppTest smoke |
| Retrieval Hit@3 / Recall@3 on labeled set | **1.000 / 1.000** (10 cases, K=3) |
| Live OpenAI grounded ask | **Not verified** — `OPENAI_API_KEY` absent |
| Manual browser UI walkthrough | **Not performed** in this environment |
| Production readiness | **Not claimed** |

---

## 2. Test results (automated)

### 2.1 Full suite (mocked OpenAI)

```text
pytest -m "not live"  →  77 passed, 1 deselected
```

Breakdown:

| Layer | Coverage | Mode |
|-------|----------|------|
| Phase 0–1 health/docs | Passed | Local FastAPI TestClient |
| Phase 2 PDF/chunk/embed/FAISS | Passed | Local + real MiniLM embeddings |
| Phase 3 `/ask` | Passed | **Mocked** OpenAI |
| Phase 4 Streamlit client/helpers/AppTest | Passed | Mocked HTTP / AppTest |
| Phase 5 edge cases (scanned/corrupt/txt/tables) | Passed | Local |
| Phase 5 E2E upload→list→select→search | Passed | Local TestClient |
| Phase 5 persistence after reload | Passed | Local FAISS disk reload |
| Phase 5 citation ↔ retrieval match | Passed | **Mocked** OpenAI |
| Phase 5 retrieval Hit@K eval | Passed | Local embeddings, no LLM |

### 2.2 Live OpenAI

```text
pytest -m live  →  1 skipped (OPENAI_API_KEY not set)
```

**Live key status:** not available in this environment → live grounded ask **unverified**.

### 2.3 Streamlit process smoke

Prior Phase 4 headless `streamlit run` startup remained green; Phase 5 reconfirmed AppTest import/smoke. **No interactive browser session was run.**

---

## 3. Retrieval quality metrics

**Dataset:** `tests/fixtures/retrieval_dataset.json` (10 questions)  
**Command:** `python scripts/eval_retrieval.py --k 3`  
**Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (real, local)

| Metric | Value |
|--------|-------|
| **Hit@3** | **1.000** (10/10) |
| **Mean Recall@3** | **1.000** |

Per-case (top pages shown):

| Case ID | Hit | Top pages @3 |
|---------|-----|----------------|
| phoenix-nest | HIT | 1, 2 |
| phoenix-year | HIT | 1, 2 |
| warehouse-code | HIT | 1, 3, 2 |
| emergency-ext | HIT | 1, 2, 3 |
| courier | HIT | 2, 1, 3 |
| hazard-form | HIT | 2, 1, 3 |
| shelf-load | HIT | 3, 1, 2 |
| widget-location | HIT | 3, 1, 2 |
| otter | HIT | 1 |
| absent-fact | HIT* | 1, 3, 2 |

\* `absent-fact` is a soft negative: success means retrieval does not fabricate CEO/salary content (not that the index is empty).

**Justified fixes this phase:** none required for Hit@K on this set (all positive cases already hit). Hardening focused on edge inputs, logging hygiene, filename sanitization, and regression coverage.

---

## 4. Real PDF / fixture validation

| Fixture | Purpose | Result |
|---------|---------|--------|
| `sample.pdf` | Known phoenix/Lake Azure fact | OK |
| `multipage_facts.pdf` | Multipage + table-like inventory text | OK |
| `second_doc.pdf` | Cross-doc / otter fact | OK |
| `blank.pdf` / `scanned_like.pdf` | No extractable text → OCR message | OK (422) |
| `corrupt.pdf` | Corrupt PDF | OK (4xx) |
| `notes.txt` | Unsupported type | OK (400) |

Single-doc, multipage, multi-doc scoped search, answer-absent soft case, and citation consistency checks are covered automatically.

---

## 5. Reliability & security review

| Area | Finding |
|------|---------|
| Upload size limit | `MAX_UPLOAD_BYTES` (default 10 MiB) enforced |
| PDF parse safety | Magic/MIME/extension checks; encrypted rejected; PyMuPDF errors mapped |
| Filename path traversal | `sanitize_filename()` strips directories |
| Timeouts / OpenAI errors | SDK timeout + retries; 503/429/502 mapping unchanged |
| Secrets / git | `.env` and `data/*` gitignored; no keys/uploads tracked |
| FAISS + metadata | Persist/reload + multi-doc consistency tested |
| Prompt injection | Untrusted excerpt wrapping + regression test retained |
| CORS | Limited to configured Streamlit localhost origins |
| Logging | Upload/search/ask latency logs **without** question bodies or API keys |

---

## 6. Frontend validation

**Automated:** Streamlit AppTest smoke; API client unit tests; E2E flow using the same upload/list/select/search contracts as the UI.

**Manual browser walkthrough:** **not executed** here.

### Exact manual steps (for a human)

1. Terminal A: `uvicorn app.main:app --reload --port 8000` (with `OPENAI_API_KEY` in `.env` for ask).
2. Terminal B: `streamlit run frontend/streamlit_app.py`.
3. Open http://localhost:8501.
4. Upload `tests/fixtures/sample.pdf` and `multipage_facts.pdf`.
5. Select “All indexed documents”, ask: “Where does the indigo phoenix nest?”
6. Select only `multipage_facts.pdf`, ask: “What is the primary warehouse code?”
7. Confirm citations show filename + page + excerpt; confirm errors if API is stopped.

---

## 7. Known limitations / outstanding issues

- Live OpenAI E2E **unverified** without a key.
- No OCR; scanned PDFs correctly rejected.
- No document DELETE API/UI.
- Hybrid BM25 **not** added (dense MiniLM Hit@3 was 1.0 on the eval set).
- Dependency versions remain ranged (not fully pinned); embedding model download is still large on first use (~MiniLM + torch stack).
- **Not production-certified** — no load test, auth, multi-tenant isolation, or deploy verification.

---

## 8. Changes delivered in Phase 5 (code)

- Expanded fixtures + `retrieval_dataset.json`
- `app/eval/retrieval.py` + `scripts/eval_retrieval.py`
- Latency logging on upload/search/ask; safer ask logging
- Filename sanitization on ingest
- New tests: edge cases, E2E integration, retrieval metrics
- This report + plan/README updates
