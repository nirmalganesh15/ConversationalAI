"""Persisted BM25 index (lexical leg of hybrid search)."""

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.core.retrieval.analyzer import Analyzer
from app.core.retrieval.models import Chunk


class BM25Index:
    def __init__(self, analyzer: Analyzer, path: Path | None = None) -> None:
        self._analyzer = analyzer
        self._path = path
        self._ids: list[str] = []
        self._owners: list[str] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        if path and path.exists():
            self._load(path)

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, chunks: list[Chunk]) -> None:
        existing = set(self._ids)
        for chunk in chunks:
            if chunk.id in existing:
                continue
            self._ids.append(chunk.id)
            self._owners.append(str(chunk.metadata.get("owner", "system")))
            self._tokens.append(self._analyzer.tokenize(chunk.text))
        self._rebuild()
        self._save()

    def remove_where(self, predicate_ids: set[str]) -> None:
        keep = [i for i, cid in enumerate(self._ids) if cid not in predicate_ids]
        self._ids = [self._ids[i] for i in keep]
        self._owners = [self._owners[i] for i in keep]
        self._tokens = [self._tokens[i] for i in keep]
        self._rebuild()
        self._save()

    def search(self, query: str, k: int, owners: set[str]) -> list[tuple[str, float]]:
        if not self._bm25:
            return []
        query_tokens = self._analyzer.tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            (
                (self._ids[i], float(s))
                for i, s in enumerate(scores)
                if s > 0 and self._owners[i] in owners
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:k]

    def _rebuild(self) -> None:
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "analyzer": self._analyzer.name,
            "ids": self._ids,
            "owners": self._owners,
            "tokens": self._tokens,
        }
        self._path.write_text(json.dumps(payload), encoding="utf-8")

    def _load(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("analyzer") != self._analyzer.name:
            return  # analyzer changed -> index is stale; it is rebuilt on next ingest
        self._ids, self._owners, self._tokens = payload["ids"], payload["owners"], payload["tokens"]
        self._rebuild()
