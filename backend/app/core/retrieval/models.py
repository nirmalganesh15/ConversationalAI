from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk plus the score trail from every retrieval stage (shown in the Inspector)."""

    chunk: Chunk
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def confidence(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf_score
