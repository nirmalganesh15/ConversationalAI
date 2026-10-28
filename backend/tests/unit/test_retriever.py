from app.core.ingestion.pipeline import SYSTEM_OWNER, KnowledgeBase
from tests.conftest import LISINOPRIL, METFORMIN, doc


def _seed(kb: KnowledgeBase) -> None:
    kb.ingest_documents(
        "docs", [doc("metformin.md", METFORMIN), doc("lisinopril.md", LISINOPRIL)], SYSTEM_OWNER
    )


def test_hybrid_rerank_finds_stemmed_match(kb: KnowledgeBase) -> None:
    _seed(kb)
    # "contraindication" (singular) only matches "contraindicated" via stemming.
    results = kb.retriever("docs").search("lisinopril contraindication", {SYSTEM_OWNER})
    assert results[0].chunk.metadata["source"] == "lisinopril.md"
    assert results[0].bm25_rank is not None
    assert results[0].rerank_score is not None


def test_every_mode_returns_results(kb: KnowledgeBase) -> None:
    _seed(kb)
    retriever = kb.retriever("docs")
    for mode in ("dense", "bm25", "hybrid", "hybrid_rerank"):
        assert retriever.search("metformin dose", {SYSTEM_OWNER}, mode=mode), mode  # type: ignore[arg-type]


def test_owner_isolation(kb: KnowledgeBase) -> None:
    kb.ingest_documents("docs", [doc("private.md", "Secret oncology protocol XYZ-123")], "alice")
    retriever = kb.retriever("docs")

    def sources(user: str) -> set[str]:
        hits = retriever.search("oncology protocol", kb.visible_owners(user))
        return {h.chunk.metadata["source"] for h in hits}

    assert "private.md" in sources("alice")
    assert "private.md" not in sources("bob")


def test_reingest_replaces_source(kb: KnowledgeBase) -> None:
    _seed(kb)
    before = len(kb.retriever("docs").scan({SYSTEM_OWNER}))
    _seed(kb)
    assert len(kb.retriever("docs").scan({SYSTEM_OWNER})) == before


def test_sources_listing(kb: KnowledgeBase) -> None:
    _seed(kb)
    sources = {s["source"] for s in kb.retriever("docs").sources({SYSTEM_OWNER})}
    assert sources == {"metformin.md", "lisinopril.md"}
