"""Shared fixtures. No network, no model downloads: hash embedder + overlap reranker +
FakeProvider make the whole graph run deterministically in milliseconds."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agents.specs import SPECS
from app.container import Container, build_container
from app.core.config import Settings
from app.core.ingestion.models import LoadedDocument, Section
from app.core.ingestion.pipeline import SYSTEM_OWNER, KnowledgeBase
from app.core.llm.fake import FakeProvider
from app.core.llm.registry import ProviderInfo, ProviderRegistry
from app.core.retrieval.models_ml import HashEmbedder, OverlapReranker
from app.core.retrieval.retriever import RetrievalParams
from app.main import create_app

METFORMIN = """# Dosage and Administration
The maximum recommended daily dose of metformin is 2550 mg in adults, given in divided doses.

# Contraindications
Metformin is contraindicated in patients with severe renal impairment (eGFR below 30 mL/min).
"""

LISINOPRIL = """# Contraindications
Lisinopril is contraindicated in patients with a history of angioedema related to previous
treatment with an ACE inhibitor, and in hypertensive patients who are pregnant.
"""

TRANSCRIPT = """[00:00:05] Dr. Rao: Welcome to the steering committee.
[00:01:10] Priya Shah: Enrollment is at 62 percent of target.
[00:02:30] Dr. Rao: We decided to open two new sites in Pune.
[00:03:45] Mark Lee: I will draft the protocol amendment by Friday.
"""


@pytest.fixture
def fake_llm() -> FakeProvider:
    return FakeProvider(answer="The maximum daily dose is 2550 mg [1].")


@pytest.fixture
def registry(fake_llm: FakeProvider) -> ProviderRegistry:
    info = ProviderInfo(
        key="fake", label="Fake", models=["fake"], default_model="fake", available=True
    )
    return ProviderRegistry({"fake": fake_llm, "claude": fake_llm}, [info])


@pytest.fixture
def kb() -> KnowledgeBase:
    return KnowledgeBase(
        agents={n.value: (s.analyzer, s.chunker) for n, s in SPECS.items()},
        embedder=HashEmbedder(),
        reranker=OverlapReranker(),
        params=RetrievalParams(top_k=4),
        data_dir=None,
    )


def doc(source: str, text: str, doc_type: str = "markdown") -> LoadedDocument:
    return LoadedDocument(
        source=source, title=source, doc_type=doc_type, sections=[Section(text=text)]
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        seed_on_startup=False,
        online_eval_enabled=True,
        min_rerank_score=0.05,
        static_dir=tmp_path / "no-static",
    )


@pytest.fixture
def container(settings: Settings, registry: ProviderRegistry) -> Container:
    c = build_container(
        settings,
        embedder=HashEmbedder(),
        reranker=OverlapReranker(),
        providers=registry,
        persist=False,
    )
    c.kb.ingest_documents(
        "docs", [doc("metformin.md", METFORMIN), doc("lisinopril.md", LISINOPRIL)], SYSTEM_OWNER
    )
    c.kb.ingest_documents("meeting", [doc("steering.txt", TRANSCRIPT, "text")], SYSTEM_OWNER)
    return c


@pytest.fixture
def client(settings: Settings, container: Container) -> Iterator[TestClient]:
    app = create_app(settings, container)
    with TestClient(app) as test_client:
        yield test_client


def login(client: TestClient, username: str, password: str) -> dict[str, str]:
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


@pytest.fixture
def admin_headers(client: TestClient) -> dict[str, str]:
    return login(client, "admin", "admin123")


@pytest.fixture
def staff_headers(client: TestClient) -> dict[str, str]:
    return login(client, "staff", "staff123")
