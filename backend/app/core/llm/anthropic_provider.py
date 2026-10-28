"""Claude adapter (official Anthropic SDK)."""

import time
from collections.abc import Callable
from typing import Any, cast

import anthropic
from anthropic.types import MessageParam, TextBlockParam

from app.core.llm.base import (
    ChatMessage,
    LLMCall,
    LLMError,
    LLMResult,
    StructuredResult,
    T,
    Tier,
    Usage,
)
from app.core.llm.pricing import estimate_cost

# Server-side refusal fallback: if the main model declines, the API re-routes in-call.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}


def _messages(messages: list[ChatMessage]) -> list[MessageParam]:
    return cast(list[MessageParam], [m.model_dump() for m in messages])


class AnthropicProvider:
    name = "claude"

    def __init__(
        self,
        api_key: str,
        main_model: str,
        fast_model: str,
        effort: str = "low",
        refusal_fallback: bool = True,
        default_max_tokens: int = 2048,
        timeout_s: float = 60.0,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout_s)
        self._models: dict[Tier, str] = {"main": main_model, "fast": fast_model}
        self._effort = effort
        self._refusal_fallback = refusal_fallback
        self._max_tokens = default_max_tokens

    def model_for(self, tier: Tier) -> str:
        return self._models[tier]

    def with_main_model(self, model: str) -> "AnthropicProvider":
        clone = object.__new__(AnthropicProvider)
        clone.__dict__.update(self.__dict__)
        clone._models = {**self._models, "main": model}
        return clone

    def _system_blocks(self, system: str) -> list[TextBlockParam]:
        # The system prompt is frozen per agent, so it is the stable cache prefix.
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    def _call(
        self, purpose: str, model: str, usage: Any, started: float, stop: str | None
    ) -> LLMCall:
        u = Usage(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
        return LLMCall(
            purpose=purpose,
            provider=self.name,
            model=model,
            usage=u,
            cost_usd=estimate_cost(model, u),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            stop_reason=stop,
        )

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
        model = self._models[tier]
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or self._max_tokens,
            "system": self._system_blocks(system),
            "messages": _messages(messages),
            "output_config": {"effort": self._effort},
        }
        use_fallback = self._refusal_fallback and model in _FALLBACK_MODELS
        stream_ctx: Any
        started = time.perf_counter()
        try:
            if use_fallback:
                stream_ctx = self._client.beta.messages.stream(
                    **params, betas=[_FALLBACK_BETA], fallbacks="default"
                )
            else:
                stream_ctx = self._client.messages.stream(**params)
            async with stream_ctx as stream:
                async for text in stream.text_stream:
                    if on_token:
                        on_token(text)
                final = await stream.get_final_message()
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Claude API") from exc

        text = "".join(b.text for b in final.content if b.type == "text")
        if final.stop_reason == "refusal":
            text = text or "The model declined to answer this request."
        return LLMResult(
            text=text,
            call=self._call(purpose, final.model, final.usage, started, final.stop_reason),
        )

    async def structured(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        schema: type[T],
        purpose: str,
        tier: Tier = "fast",
    ) -> StructuredResult[T]:
        model = self._models[tier]
        started = time.perf_counter()
        try:
            response = await self._client.messages.parse(
                model=model,
                max_tokens=4096,
                system=self._system_blocks(system),
                messages=_messages(messages),
                output_format=schema,
            )
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Claude API") from exc
        if response.parsed_output is None:
            raise LLMError(f"Structured output missing (stop_reason={response.stop_reason})")
        return StructuredResult(
            data=response.parsed_output,
            call=self._call(purpose, model, response.usage, started, response.stop_reason),
        )
