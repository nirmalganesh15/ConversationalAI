"""KnowledgeBase: one hybrid retriever per agent + the ingestion entry points."""

import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from app.core.ingestion.chunkers import Chunker
from app.core.ingestion.loaders import is_code_path, load_bytes, load_code_file, load_code_zip
from app.core.ingestion.models import LoadedDocument
from app.core.observability.logging import get_logger
from app.core.retrieval.analyzer import Analyzer
from app.core.retrieval.bm25 import BM25Index
from app.core.retrieval.models_ml import Embedder, Reranker
from app.core.retrieval.retriever import HybridRetriever, RetrievalParams
from app.core.retrieval.vector_store import VectorStore, make_chroma_client

log = get_logger(__name__)

SYSTEM_OWNER = "system"  # seeded corpora, visible to every user of the agent
_GITHUB_URL = re.compile(r"^https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")
MAX_ZIP_BYTES = 60 * 1024 * 1024


class IngestReport(BaseModel):
    agent: str
    documents: int
    chunks: int
    sources: list[str]
    skipped: list[str] = []


class StoredDocument(BaseModel):
    source: str
    title: str
    owner: str
    text: str
    metadata: dict[str, Any] = {}


class DocumentStore:
    """Full-text copy of each ingested document, per agent.

    Chunks are for retrieval; some tools need the whole document (gap analysis over a
    full resume, action-item extraction over a full transcript).
    """

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._docs: dict[str, StoredDocument] = {}
        if path and path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._docs = {k: StoredDocument(**v) for k, v in raw.items()}

    @staticmethod
    def _key(owner: str, source: str) -> str:
        return f"{owner}::{source}"

    def put(self, doc: StoredDocument) -> None:
        with self._lock:
            self._docs[self._key(doc.owner, doc.source)] = doc
            self._save()

    def delete(self, owner: str, source: str) -> None:
        with self._lock:
            self._docs.pop(self._key(owner, source), None)
            self._save()

    def list(self, owners: set[str], **filters: str) -> list[StoredDocument]:
        docs = [
            d
            for d in self._docs.values()
            if d.owner in owners and all(d.metadata.get(k) == v for k, v in filters.items())
        ]
        return sorted(docs, key=lambda d: d.source)

    def _save(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {k: v.model_dump() for k, v in self._docs.items()}
            self._path.write_text(json.dumps(payload), encoding="utf-8")


class KnowledgeBase:
    def __init__(
        self,
        agents: dict[str, tuple[Analyzer, Chunker]],
        embedder: Embedder,
        reranker: Reranker | None,
        params: RetrievalParams,
        data_dir: Path | None,
    ) -> None:
        client = make_chroma_client(data_dir / "chroma" if data_dir else None)
        # In-memory Chroma clients share state per process; isolate ephemeral KBs.
        prefix = "atlas" if data_dir else f"atlas_{uuid.uuid4().hex[:8]}"
        self._chunkers: dict[str, Chunker] = {}
        self._retrievers: dict[str, HybridRetriever] = {}
        self._documents: dict[str, DocumentStore] = {}
        for agent, (analyzer, chunker) in agents.items():
            self._documents[agent] = DocumentStore(
                data_dir / "documents" / f"{agent}.json" if data_dir else None
            )
            bm25_path = data_dir / "bm25" / f"{agent}.json" if data_dir else None
            self._chunkers[agent] = chunker
            self._retrievers[agent] = HybridRetriever(
                store=VectorStore(client, f"{prefix}_{agent}"),
                bm25=BM25Index(analyzer, bm25_path),
                embedder=embedder,
                reranker=reranker,
                params=params,
            )

    def retriever(self, agent: str) -> HybridRetriever:
        return self._retrievers[agent]

    def documents(self, agent: str) -> DocumentStore:
        return self._documents[agent]

    def delete_source(self, agent: str, source: str, owner: str) -> int:
        self._documents[agent].delete(owner, source)
        return self._retrievers[agent].delete_where(
            {"$and": [{"source": source}, {"owner": owner}]}
        )

    @staticmethod
    def visible_owners(user_id: str) -> set[str]:
        return {SYSTEM_OWNER, user_id}

    # --- ingestion ---------------------------------------------------------------------
    def ingest_documents(
        self,
        agent: str,
        docs: list[LoadedDocument],
        owner: str,
        extra: dict[str, Any] | None = None,
    ) -> IngestReport:
        retriever = self._retrievers[agent]
        chunker = self._chunkers[agent]
        total = 0
        store = self._documents[agent]
        for doc in docs:
            # Re-ingesting a source replaces it (idempotent seeding and uploads).
            retriever.delete_where({"$and": [{"source": doc.source}, {"owner": owner}]})
            meta = {"agent": agent, **(extra or {})}
            chunks = chunker.chunk(doc, owner, meta)
            total += retriever.add(chunks)
            if agent != "code":  # code docs are retrieved by symbol, not read whole
                store.put(
                    StoredDocument(
                        source=doc.source,
                        title=doc.title,
                        owner=owner,
                        text=doc.full_text,
                        metadata={k: str(v) for k, v in {**doc.metadata, **meta}.items()},
                    )
                )
        log.info("ingested", agent=agent, owner=owner, documents=len(docs), chunks=total)
        return IngestReport(
            agent=agent, documents=len(docs), chunks=total, sources=[d.source for d in docs]
        )

    def ingest_file(
        self,
        agent: str,
        filename: str,
        data: bytes,
        owner: str,
        extra: dict[str, Any] | None = None,
    ) -> IngestReport:
        if agent == "code":
            if filename.lower().endswith(".zip"):
                docs = load_code_zip(data, repo=Path(filename).stem)
            else:
                doc = load_code_file(filename, data, repo="upload")
                docs = [doc] if doc else []
            return self.ingest_documents(agent, docs, owner, extra)
        return self.ingest_documents(agent, [load_bytes(filename, data)], owner, extra)

    def ingest_directory(
        self, agent: str, root: Path, owner: str, repo: str, include: list[str]
    ) -> IngestReport:
        """Index selected paths of a local source tree (used to self-ingest this repo).

        Replaces everything previously indexed for `repo`, so deleted files disappear too.
        """
        self._retrievers[agent].delete_where({"$and": [{"owner": owner}, {"repo": repo}]})
        docs: list[LoadedDocument] = []
        candidates = sorted(
            p
            for item in include
            if (root / item).exists()
            for p in ([root / item] if (root / item).is_file() else (root / item).rglob("*"))
        )
        for path in candidates:
            rel = path.relative_to(root).as_posix()
            if path.is_file() and is_code_path(rel):
                doc = load_code_file(rel, path.read_bytes(), repo)
                if doc:
                    docs.append(doc)
        return self.ingest_documents(agent, docs, owner)

    async def ingest_github(self, url: str, owner: str) -> IngestReport:
        match = _GITHUB_URL.match(url.strip())
        if not match:
            raise ValueError("Expected a URL like https://github.com/<owner>/<repo>")
        gh_owner, repo = match.groups()
        archive = f"https://codeload.github.com/{gh_owner}/{repo}/zip/HEAD"
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(archive)
        if response.status_code != 200:
            raise ValueError(f"Could not download {gh_owner}/{repo} (HTTP {response.status_code})")
        if len(response.content) > MAX_ZIP_BYTES:
            raise ValueError("Repository archive is larger than 60 MB")
        docs = load_code_zip(response.content, repo=f"{gh_owner}/{repo}")
        return self.ingest_documents("code", docs, owner, {"repo": f"{gh_owner}/{repo}"})
