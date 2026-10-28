"""Provider-neutral LLM port. Agents depend on this, never on a vendor SDK."""

from collections.abc import Callable
from typing import Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, Field

Tier = Literal["main", "fast"]
T = TypeVar("T", bound=BaseModel)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class LLMCall(BaseModel):
    """One model invocation - what the Inspector 'Usage' tab renders, row by row."""

    purpose: str
    provider: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    stop_reason: str | None = None


class LLMResult(BaseModel):
    text: str
    call: LLMCall


class StructuredResult(BaseModel, Generic[T]):  # noqa: UP046
    data: T
    call: LLMCall


class LLMError(RuntimeError):
    """Raised for provider failures the graph should surface to the user cleanly."""


class LLMProvider(Protocol):
    name: str

    def model_for(self, tier: Tier) -> str: ...

    async def complete(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        purpose: str,
        tier: Tier = "main",
        max_tokens: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResult:
        """Generate text. If on_token is given the response is streamed through it."""
        ...

    async def structured(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        schema: type[T],
        purpose: str,
        tier: Tier = "fast",
    ) -> StructuredResult[T]:
        """Generate a response validated against a Pydantic schema."""
        ...
