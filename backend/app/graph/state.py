"""Graph state. Persisted per thread (user:session) by the LangGraph checkpointer.

Two kinds of fields:
- memory, carried across turns: `messages` (LLM working memory, compacted), `summary`,
  and `transcript` (full display history with per-turn metadata for the UI);
- per-turn scratch, reset by the API on every new question.
"""

from dataclasses import dataclass
from typing import Any, TypedDict

from app.core.config import Settings
from app.core.ingestion.pipeline import KnowledgeBase
from app.core.llm.registry import ProviderRegistry


class ChatState(TypedDict, total=False):
    # who / what (per turn)
    user_id: str
    role: str
    agent: str
    provider: str
    model: str
    question: str
    query: str  # question after input guardrails (e.g. PII redacted)

    # memory (persisted)
    messages: list[dict[str, str]]
    summary: str
    transcript: list[dict[str, Any]]

    # per-turn scratch
    search_query: str
    planned_tools: list[dict[str, str]]
    tool_outputs: list[dict[str, str]]
    contexts: list[dict[str, Any]]
    answer: str
    guardrails: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]
    timings: dict[str, float]
    evals: dict[str, Any] | None
    blocked: bool
    flags: list[str]


def new_turn(
    *, user_id: str, role: str, agent: str, provider: str, model: str, question: str
) -> ChatState:
    """Input for one graph run: identity + question, with all scratch fields reset."""
    return ChatState(
        user_id=user_id,
        role=role,
        agent=agent,
        provider=provider,
        model=model,
        question=question,
        query=question,
        search_query="",
        planned_tools=[],
        tool_outputs=[],
        contexts=[],
        answer="",
        guardrails=[],
        llm_calls=[],
        timings={},
        evals=None,
        blocked=False,
        flags=[],
    )


@dataclass
class GraphDeps:
    kb: KnowledgeBase
    providers: ProviderRegistry
    settings: Settings
