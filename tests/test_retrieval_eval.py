"""Offline retrieval quality: Hit@K / Recall@K on labeled fixture questions."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.eval.retrieval import load_dataset, run_retrieval_eval
from app.services.embeddings import get_embedding_service
from app.services.ingest import IngestService
from app.services.vector_store import VectorStore

FIXTURES = Path(__file__).parent / "fixtures"
DATASET = FIXTURES / "retrieval_dataset.json"


@pytest.fixture(scope="module")
def retrieval_summary(tmp_path_factory):
    data_root = tmp_path_factory.mktemp("retrieval-eval")
    get_settings.cache_clear()
    settings = Settings(
        data_dir=str(data_root),
        chunk_size=400,
        chunk_overlap=80,
        top_k=5,
        similarity_threshold=0.0,
    )
    embeddings = get_embedding_service(settings.embedding_model)
    store = VectorStore(settings.data_dir, dimension=embeddings.dimension)
    ingest = IngestService(settings, embeddings, store)

    cases = load_dataset(DATASET)
    doc_ids: dict[str, str] = {}
    for case in cases:
        name = case["fixture"]
        if name in doc_ids:
            continue
        pdf = (FIXTURES / name).read_bytes()
        result = ingest.ingest_pdf(pdf, filename=name, content_type="application/pdf")
        doc_ids[name] = result.document_id

    summary = run_retrieval_eval(ingest, cases, doc_id_by_fixture=doc_ids, k=3)
    get_settings.cache_clear()
    return summary


def test_hit_at_3_meets_threshold(retrieval_summary) -> None:
    # Positive + soft negative cases; require strong overall hit rate.
    assert retrieval_summary.n_cases >= 8
    assert retrieval_summary.hit_at_k >= 0.8, [
        (r.case_id, r.hit_at_k, r.top_pages) for r in retrieval_summary.results
    ]


def test_mean_recall_at_3(retrieval_summary) -> None:
    assert retrieval_summary.mean_recall_at_k >= 0.75


def test_positive_cases_hit_expected_page_or_phrase(retrieval_summary) -> None:
    misses = [
        r
        for r in retrieval_summary.results
        if r.case_id != "absent-fact" and not r.hit_at_k
    ]
    assert misses == [], [(m.case_id, m.top_pages) for m in misses]
