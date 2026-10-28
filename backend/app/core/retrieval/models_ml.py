"""Embedding + reranking models behind small protocols so tests can swap them out."""

import hashlib
import math
from functools import cached_property
from typing import Any, Protocol

# bge models are trained with this instruction prefix on the query side only.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    @cached_property
    def _model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self._model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        prefix = _BGE_QUERY_PREFIX if "bge" in self._model_name.lower() else ""
        vector = self._model.encode([prefix + text], normalize_embeddings=True)[0]
        return list(vector.tolist())


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    @cached_property
    def _model(self) -> Any:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self._model_name, device="cpu", max_length=512)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        import torch

        # Force a sigmoid explicitly: models differ in their default activation (bge applies
        # one, ms-marco returns logits), and the confidence threshold needs one 0..1 scale.
        scores = self._model.predict(
            [(query, p) for p in passages], activation_fn=torch.nn.Sigmoid()
        )
        return [float(s) for s in scores]


class HashEmbedder:
    """Bag-of-words hashing embedder - deterministic, dependency-free, for tests only."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for word in text.lower().split():
            digest = hashlib.md5(word.strip(".,?!:;()").encode(), usedforsecurity=False)
            vec[int(digest.hexdigest(), 16) % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OverlapReranker:
    """Token-overlap reranker for tests (no model download)."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(query.lower().split())
        return [len(q & set(p.lower().split())) / (len(q) or 1) for p in passages]
