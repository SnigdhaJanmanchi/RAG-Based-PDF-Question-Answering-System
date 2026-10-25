"""Retrieval quality metrics: Hit@K / Recall@K over a small labeled dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.ingest import IngestService
from app.services.vector_store import SearchHit


@dataclass(frozen=True)
class EvalCaseResult:
    case_id: str
    question: str
    hit_at_k: bool
    recall_at_k: float
    top_pages: list[int]
    notes: str = ""


@dataclass(frozen=True)
class EvalSummary:
    k: int
    n_cases: int
    hit_at_k: float
    mean_recall_at_k: float
    results: list[EvalCaseResult]


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _phrase_hit(hits: list[SearchHit], phrases: list[str]) -> bool:
    if not phrases:
        return False
    blob = " ".join(h.text for h in hits).lower()
    return any(p.lower() in blob for p in phrases)


def _page_hit(hits: list[SearchHit], expected_pages: list[int]) -> bool:
    if not expected_pages:
        return False
    pages = {h.page for h in hits}
    return any(p in pages for p in expected_pages)


def score_case(
    hits: list[SearchHit],
    *,
    expected_pages: list[int],
    expected_phrases: list[str],
    expect_no_strong_hit: bool = False,
) -> tuple[bool, float, list[int]]:
    """Return (hit@k, recall@k, top_pages).

    For positive cases: Hit@K if any top-K hit matches an expected page OR phrase.
    Recall@K = fraction of expected pages that appear in top-K (1.0 if no expected pages
    and expect_no_strong_hit and no phrase leakage of inventable CEO-style content).
    """
    top_pages = [h.page for h in hits]

    if expect_no_strong_hit:
        # Soft check: do not require empty retrieval; require that none of the
        # forbidden invented markers appear. Treat as hit if retrieval is empty
        # or texts lack salary/CEO fabrication cues from the question domain.
        blob = " ".join(h.text for h in hits).lower()
        fabricated = ("ceo" in blob and "salary" in blob) or ("$" in blob and "salary" in blob)
        ok = not fabricated
        return ok, 1.0 if ok else 0.0, top_pages

    page_ok = _page_hit(hits, expected_pages)
    phrase_ok = _phrase_hit(hits, expected_phrases)
    hit = page_ok or phrase_ok

    if not expected_pages:
        recall = 1.0 if hit else 0.0
    else:
        found = {h.page for h in hits}
        recall = len(found.intersection(expected_pages)) / len(set(expected_pages))

    return hit, recall, top_pages


def run_retrieval_eval(
    ingest: IngestService,
    cases: list[dict[str, Any]],
    *,
    doc_id_by_fixture: dict[str, str],
    k: int = 3,
) -> EvalSummary:
    results: list[EvalCaseResult] = []
    for case in cases:
        fixture = case["fixture"]
        doc_id = doc_id_by_fixture[fixture]
        question = case["question"]
        hits = ingest.search(question, top_k=k, doc_id=doc_id)
        hit, recall, top_pages = score_case(
            hits,
            expected_pages=list(case.get("expected_pages") or []),
            expected_phrases=list(case.get("expected_phrases") or []),
            expect_no_strong_hit=bool(case.get("expect_no_strong_hit")),
        )
        results.append(
            EvalCaseResult(
                case_id=str(case["id"]),
                question=question,
                hit_at_k=hit,
                recall_at_k=recall,
                top_pages=top_pages,
                notes=str(case.get("notes") or ""),
            )
        )

    n = len(results) or 1
    hit_rate = sum(1 for r in results if r.hit_at_k) / n
    mean_recall = sum(r.recall_at_k for r in results) / n
    return EvalSummary(
        k=k,
        n_cases=len(results),
        hit_at_k=hit_rate,
        mean_recall_at_k=mean_recall,
        results=results,
    )
