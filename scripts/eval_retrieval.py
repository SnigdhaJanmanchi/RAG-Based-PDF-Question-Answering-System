#!/usr/bin/env python3
"""Run offline retrieval Hit@K / Recall@K on fixture PDFs (no OpenAI)."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.eval.retrieval import load_dataset, run_retrieval_eval
from app.services.embeddings import get_embedding_service
from app.services.ingest import IngestService
from app.services.vector_store import VectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "retrieval_dataset.json",
    )
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    fixtures_dir = args.dataset.parent

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            data_dir=tmp,
            chunk_size=400,
            chunk_overlap=80,
            top_k=args.k,
            similarity_threshold=0.0,  # score all hits for offline eval
        )
        embeddings = get_embedding_service(settings.embedding_model)
        store = VectorStore(settings.data_dir, dimension=embeddings.dimension)
        ingest = IngestService(settings, embeddings, store)

        doc_id_by_fixture: dict[str, str] = {}
        for case in cases:
            name = case["fixture"]
            if name in doc_id_by_fixture:
                continue
            pdf_path = fixtures_dir / name
            result = ingest.ingest_pdf(
                pdf_path.read_bytes(),
                filename=name,
                content_type="application/pdf",
            )
            doc_id_by_fixture[name] = result.document_id

        summary = run_retrieval_eval(
            ingest, cases, doc_id_by_fixture=doc_id_by_fixture, k=args.k
        )

    print(f"Cases: {summary.n_cases}  K={summary.k}")
    print(f"Hit@{summary.k}: {summary.hit_at_k:.3f}")
    print(f"Mean Recall@{summary.k}: {summary.mean_recall_at_k:.3f}")
    for r in summary.results:
        mark = "HIT" if r.hit_at_k else "MISS"
        print(f"  [{mark}] {r.case_id} pages={r.top_pages} recall={r.recall_at_k:.2f}")
    return 0 if summary.hit_at_k >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
