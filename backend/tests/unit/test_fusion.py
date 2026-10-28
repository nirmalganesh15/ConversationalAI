from app.core.retrieval.fusion import reciprocal_rank_fusion


def test_document_ranked_well_in_both_lists_wins() -> None:
    fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"], "bm25": ["b", "d", "a"]})
    assert [doc_id for doc_id, _, _ in fused][:2] == ["b", "a"]


def test_ranks_are_reported_per_leg() -> None:
    fused = dict((d, r) for d, _, r in reciprocal_rank_fusion({"dense": ["a"], "bm25": ["b", "a"]}))
    assert fused["a"] == {"dense": 1, "bm25": 2}
    assert fused["b"] == {"bm25": 1}


def test_ties_are_deterministic() -> None:
    first = reciprocal_rank_fusion({"dense": ["x"], "bm25": ["y"]})
    second = reciprocal_rank_fusion({"bm25": ["y"], "dense": ["x"]})
    assert first == second


def test_empty_rankings() -> None:
    assert reciprocal_rank_fusion({}) == []
