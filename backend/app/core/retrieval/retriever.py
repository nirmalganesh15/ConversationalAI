"""Hybrid retriever: BM25 (stemmed) + dense -> RRF -> cross-encoder rerank.

`mode` exists so the offline eval can run the ablation (dense / bm25 / hybrid /
hybrid_rerank) through exactly the same code path the app uses.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from app.core.retrieval.bm25 import BM25Index
from app.core.retrieval.fusion import reciprocal_rank_fusion
from app.core.retrieval.models import Chunk, RetrievedChunk
from app.core.retrieval.models_ml import Embedder, Reranker
from app.core.retrieval.vector_store import VectorStore

RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


@dataclass(frozen=True)
class RetrievalParams:
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    rerank_pool: int = 30
    top_k: int = 6


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        bm25: BM25Index,
        embedder: Embedder,
        reranker: Reranker | None,
        params: RetrievalParams,
    ) -> None:
        self._store = store
        self._bm25 = bm25
        self._embedder = embedder
        self._reranker = reranker
        self._params = params

    # --- indexing --------------------------------------------------------------
    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        embeddings = self._embedder.embed_documents([c.text for c in chunks])
        self._store.upsert(chunks, embeddings)
        self._bm25.add(chunks)
        return len(chunks)

    def delete_where(self, where: dict[str, Any]) -> int:
        ids = self._store.ids_where(where)
        self._store.delete(ids)
        self._bm25.remove_where(set(ids))
        return len(ids)

    def sources(self, owners: set[str]) -> list[dict[str, Any]]:
        """One row per ingested source document, with its chunk count."""
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for meta in self._store.all_metadata(owners):
            key = (str(meta.get("source")), str(meta.get("owner")))
            counts[key] += 1
            grouped.setdefault(
                key,
                {
                    "source": meta.get("source"),
                    "title": meta.get("title", meta.get("source")),
                    "owner": meta.get("owner"),
                    "doc_type": meta.get("doc_type", ""),
                },
            )
        return [{**row, "chunks": counts[key]} for key, row in sorted(grouped.items())]

    def is_empty(self) -> bool:
        return self._store.count() == 0

    # --- search ----------------------------------------------------------------
    def search(
        self,
        query: str,
        owners: set[str],
        mode: RetrievalMode = "hybrid_rerank",
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        p = self._params
        top_k = top_k or p.top_k

        rankings: dict[str, list[str]] = {}
        if mode in ("dense", "hybrid", "hybrid_rerank"):
            dense = self._store.query(self._embedder.embed_query(query), p.dense_top_k, owners)
            rankings["dense"] = [cid for cid, _ in dense]
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            sparse = self._bm25.search(query, p.sparse_top_k, owners)
            rankings["bm25"] = [cid for cid, _ in sparse]

        fused = reciprocal_rank_fusion(rankings, k=p.rrf_k)
        pool_size = p.rerank_pool if mode == "hybrid_rerank" else top_k
        pool = fused[:pool_size]
        chunks = self._store.get([cid for cid, _, _ in pool])

        results = [
            RetrievedChunk(
                chunk=chunks[cid],
                bm25_rank=ranks.get("bm25"),
                dense_rank=ranks.get("dense"),
                rrf_score=score,
            )
            for cid, score, ranks in pool
            if cid in chunks
        ]

        if mode == "hybrid_rerank" and self._reranker and results:
            scores = self._reranker.score(query, [r.chunk.text for r in results])
            for result, score in zip(results, scores, strict=True):
                result.rerank_score = round(score, 4)
            results.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)

        return results[:top_k]

    def scan(self, owners: set[str]) -> list[Chunk]:
        return self._store.scan(owners)
