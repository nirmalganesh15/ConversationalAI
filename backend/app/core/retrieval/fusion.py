"""Reciprocal Rank Fusion.

RRF only uses ranks, so it fuses BM25 scores (unbounded) and cosine similarities
(bounded) without any score normalisation - the main reason it was chosen over a
weighted linear blend.
"""


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]], k: int = 60
) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse ranked id lists. Returns (id, score, {ranking_name: 1-based rank}) best-first.

    Ties are broken by id so results are deterministic.
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ids in rankings.items():
        for position, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)
            ranks.setdefault(doc_id, {})[name] = position
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [(doc_id, round(score, 6), ranks[doc_id]) for doc_id, score in ordered]
