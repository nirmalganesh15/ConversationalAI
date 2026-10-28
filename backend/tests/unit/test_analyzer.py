from app.core.retrieval.analyzer import CodeAnalyzer, StemmingAnalyzer


def test_stemming_normalises_morphological_variants() -> None:
    a = StemmingAnalyzer()
    assert a.tokenize("diagnosed") == a.tokenize("diagnosing")
    assert a.tokenize("hypertension") == a.tokenize("hypertensive")
    assert a.tokenize("contraindicated") == a.tokenize("contraindications")


def test_known_stemming_gap_is_covered_by_dense_leg() -> None:
    # Snowball maps "diagnosis" -> "diagnosi" but "diagnosed" -> "diagnos". Documented
    # limitation: this is exactly the kind of miss the embedding leg of hybrid search covers.
    a = StemmingAnalyzer()
    assert a.tokenize("diagnosis") != a.tokenize("diagnosed")


def test_stopwords_removed() -> None:
    assert StemmingAnalyzer().tokenize("what is the dose of the drug") == ["dose", "drug"]


def test_clinical_abbreviations_kept_verbatim() -> None:
    tokens = StemmingAnalyzer().tokenize("Check HbA1c and eGFR in T2DM patients on ACE inhibitors")
    assert {"hba1c", "egfr", "t2dm", "ace"} <= set(tokens)


def test_hyphenated_compounds_split() -> None:
    tokens = StemmingAnalyzer().tokenize("beta-blockers")
    assert "beta" in tokens and "blocker" in tokens


def test_code_analyzer_splits_identifiers() -> None:
    tokens = CodeAnalyzer().tokenize("class HybridRetriever: def search_chunks(self)")
    assert {"hybridretriever", "hybrid", "retriever", "search_chunks", "search", "chunks"} <= set(
        tokens
    )
