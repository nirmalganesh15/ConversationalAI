"""Composition root: builds every long-lived object once, in one place."""

from dataclasses import dataclass, field
from typing import Any

from app.agents.specs import SPECS
from app.core.config import Settings
from app.core.db import Database
from app.core.guardrails.checks import RateLimiter
from app.core.ingestion.pipeline import KnowledgeBase
from app.core.llm.registry import ProviderRegistry
from app.core.retrieval.models_ml import (
    CrossEncoderReranker,
    Embedder,
    Reranker,
    SentenceTransformerEmbedder,
)
from app.core.retrieval.retriever import RetrievalParams
from app.core.security.policy import Role


@dataclass
class Container:
    settings: Settings
    db: Database
    kb: KnowledgeBase
    providers: ProviderRegistry
    rate_limiter: RateLimiter
    graph: Any = None
    status: dict[str, str] = field(default_factory=lambda: {"seed": "pending"})


def build_container(
    settings: Settings,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    providers: ProviderRegistry | None = None,
    persist: bool = True,
) -> Container:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(settings.sqlite_path)
    db.upsert_user(settings.admin_username, settings.admin_password.get_secret_value(), Role.ADMIN)
    db.upsert_user(settings.staff_username, settings.staff_password.get_secret_value(), Role.STAFF)

    if reranker is None and settings.reranker_enabled:
        reranker = CrossEncoderReranker(settings.reranker_model)
    kb = KnowledgeBase(
        agents={name.value: (spec.analyzer, spec.chunker) for name, spec in SPECS.items()},
        embedder=embedder or SentenceTransformerEmbedder(settings.embedding_model),
        reranker=reranker if settings.reranker_enabled else None,
        params=RetrievalParams(
            dense_top_k=settings.dense_top_k,
            sparse_top_k=settings.sparse_top_k,
            rrf_k=settings.rrf_k,
            rerank_pool=settings.rerank_pool,
            top_k=settings.context_top_k,
        ),
        data_dir=settings.data_dir if persist else None,
    )
    return Container(
        settings=settings,
        db=db,
        kb=kb,
        providers=providers or ProviderRegistry.from_settings(settings),
        rate_limiter=RateLimiter(settings.rate_limit_per_minute),
    )
