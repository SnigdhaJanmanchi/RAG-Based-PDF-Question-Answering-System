"""Overview / summary intent detection and document-scoped overview retrieval.

Used for questions like "what is this about", "summarize", "key findings",
or "aim of the project". Does NOT apply to arbitrary factual questions.
"""

from __future__ import annotations

import re
from typing import Iterable

from app.services.vector_store import SearchHit

# Explicit summary / synthesize requests — prefer document overview over similarity.
_EXPLICIT_SUMMARY_RE = re.compile(
    r"(?is)\b("
    r"summarize|summarise|summary|overview|"
    r"key\s+findings?|main\s+(?:points?|ideas?|takeaways?)|"
    r"tl;?dr|tldr"
    r")\b"
)

# Broader topic / aim / "what is this about" questions (summary patterns included
# via is_overview_question combining both checks).
_TOPIC_OVERVIEW_RE = re.compile(
    r"(?is)\b("
    r"what\s+is\s+(?:this|the)\s+(?:document|pdf|paper|file|project)\s+about|"
    r"what\s+(?:is|are)\s+the\s+topic|"
    r"what\s+is\s+the\s+main\s+topic|"
    r"topic\s+of\s+(?:this|the)\s+(?:document|pdf|paper|file)|"
    r"about\s+(?:this|the)\s+(?:document|pdf|paper|file)|"
    r"aim\s+of\s+(?:the\s+)?(?:project|document|paper|study)|"
    r"purpose\s+of\s+(?:this|the)\s+(?:document|pdf|paper|project|study)|"
    r"what\s+does\s+(?:this|the)\s+(?:document|pdf|paper)\s+"
    r"(?:discuss|cover|describe|say)|"
    r"main\s+(?:theme|subject|focus)|"
    r"give\s+(?:me\s+)?(?:an?\s+)?overview"
    r")\b"
)


def is_explicit_summary_request(question: str) -> bool:
    """True when the user explicitly asks for a summary / key findings / overview."""
    q = (question or "").strip()
    if not q:
        return False
    return _EXPLICIT_SUMMARY_RE.search(q) is not None


def is_overview_question(question: str) -> bool:
    """True for topic/about/summary/aim-style questions (includes explicit summaries)."""
    q = (question or "").strip()
    if not q:
        return False
    if is_explicit_summary_request(q):
        return True
    return _TOPIC_OVERVIEW_RE.search(q) is not None


def max_hit_score(hits: Iterable[SearchHit]) -> float:
    scores = [h.score for h in hits]
    return max(scores) if scores else 0.0


def should_use_small_doc_overview(
    *,
    is_overview: bool,
    chunk_count: int,
    small_doc_max_chunks: int,
    semantic_hits: list[SearchHit],
    similarity_threshold: float,
    weak_score_floor: float,
    document_scoped: bool,
) -> bool:
    """Conservative overview path for small docs when similarity is too weak.

    Only for overview/topic questions — never for arbitrary factual questions.

    - With an explicit ``document_id`` (or a single-doc library): allow overview
      chunks whenever the main threshold produced no hits. Meta-questions like
      "what is the topic" often score near zero / negative against content.
    - Without a document scope: require a weak-but-usable max semantic score
      ``>= weak_score_floor`` and ``< similarity_threshold`` so we do not dump
      leading chunks from an unrelated multi-doc corpus.
    """
    if not is_overview:
        return False
    if chunk_count <= 0 or chunk_count > small_doc_max_chunks:
        return False
    if similarity_threshold <= 0:
        return False
    if document_scoped:
        return True
    if weak_score_floor >= similarity_threshold:
        return False
    best = max_hit_score(semantic_hits)
    return weak_score_floor <= best < similarity_threshold


def merge_hits_prefer_order(
    preferred: list[SearchHit],
    extra: list[SearchHit],
    *,
    limit: int,
) -> list[SearchHit]:
    """Merge hits, keeping ``preferred`` first and deduping by chunk_id."""
    seen: set[str] = set()
    out: list[SearchHit] = []
    for hit in preferred + extra:
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        out.append(hit)
        if len(out) >= limit:
            break
    return out
