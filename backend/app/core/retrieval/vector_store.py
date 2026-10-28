"""ChromaDB wrapper (dense leg). Embeddings are computed by us, not by Chroma, so the
embedding model is explicit config and identical at index and query time."""

from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.api.types import Where
from chromadb.config import Settings as ChromaSettings

from app.core.retrieval.models import Chunk


def make_chroma_client(path: Path | None) -> chromadb.ClientAPI:
    settings = ChromaSettings(anonymized_telemetry=False, allow_reset=True)
    if path is None:
        return chromadb.EphemeralClient(settings=settings)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=settings)


class VectorStore:
    def __init__(self, client: chromadb.ClientAPI, collection: str) -> None:
        self._collection = client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._collection.count()

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for start in range(0, len(chunks), 256):
            batch = chunks[start : start + 256]
            self._collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[_clean_metadata(c.metadata) for c in batch],
                embeddings=embeddings[start : start + 256],  # type: ignore[arg-type]
            )

    def query(self, embedding: list[float], k: int, owners: set[str]) -> list[tuple[str, float]]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=min(k, self.count()),
            where=_owner_filter(owners),
        )
        ids = result["ids"][0]
        distances = (result.get("distances") or [[0.0] * len(ids)])[0]
        return [(i, 1.0 - float(d)) for i, d in zip(ids, distances, strict=True)]

    def get(self, ids: list[str]) -> dict[str, Chunk]:
        if not ids:
            return {}
        result = self._collection.get(ids=ids)
        metadatas = result.get("metadatas") or [{}] * len(result["ids"])
        documents = result.get("documents") or [""] * len(result["ids"])
        return {
            cid: Chunk(id=cid, text=doc or "", metadata=dict(meta or {}))
            for cid, doc, meta in zip(result["ids"], documents, metadatas, strict=True)
        }

    def ids_where(self, where: dict[str, Any]) -> list[str]:
        return list(self._collection.get(where=where, include=[])["ids"])

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._collection.delete(ids=ids)

    def all_metadata(self, owners: set[str]) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []
        result = self._collection.get(where=_owner_filter(owners), include=["metadatas"])
        return [dict(m or {}) for m in result.get("metadatas") or []]

    def scan(self, owners: set[str]) -> list[Chunk]:
        """Full scan for whole-corpus tools (endpoint listing etc.). Fine at demo scale;
        in production this becomes a metadata query against Postgres."""
        if self.count() == 0:
            return []
        result = self._collection.get(
            where=_owner_filter(owners), include=["documents", "metadatas"]
        )
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return [
            Chunk(id=cid, text=doc or "", metadata=dict(meta or {}))
            for cid, doc, meta in zip(result["ids"], documents, metadatas, strict=True)
        ]


def _owner_filter(owners: set[str]) -> Where:
    return cast(Where, {"owner": {"$in": sorted(owners)}})


def _clean_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Chroma only stores scalars; lists are joined so they stay filterable/readable."""
    clean: dict[str, str | int | float | bool] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, list | tuple | set):
            clean[key] = ", ".join(map(str, value))
        elif isinstance(value, str | int | float | bool):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean
