"""Open-weight adapter for any OpenAI-compatible endpoint.

Default target is Hugging Face Inference Providers (router.huggingface.co/v1); the same
code talks to Ollama (http://localhost:11434/v1), Groq, OpenRouter or vLLM by env change.
"""

import json
import re
import time
from collections.abc import Callable
from typing import Any, cast

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError

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

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class OpenWeightProvider:
    name = "open"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        default_max_tokens: int = 2048,
        timeout_s: float = 60.0,
    ) -> None:
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self._model = model
        self._max_tokens = default_max_tokens

    def model_for(self, tier: Tier) -> str:
        # One open model serves both tiers; a smaller "fast" model is a config change away.
        return self._model

    def with_main_model(self, model: str) -> "OpenWeightProvider":
        clone = object.__new__(OpenWeightProvider)
        clone.__dict__.update(self.__dict__)
        clone._model = model
        return clone

    @staticmethod
    def _messages(system: str, messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
        rows = [{"role": "system", "content": system}, *(m.model_dump() for m in messages)]
        return cast(list[ChatCompletionMessageParam], rows)

    def _call(self, purpose: str, usage: Any, started: float, stop: str | None) -> LLMCall:
        u = Usage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        return LLMCall(
            purpose=purpose,
            provider=self.name,
            model=self._model,
            usage=u,
            cost_usd=estimate_cost(self._model, u),
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
        started = time.perf_counter()
        parts: list[str] = []
        usage: Any = None
        stop: str | None = None
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=self._messages(system, messages),
                max_tokens=max_tokens or self._max_tokens,
                temperature=0.2,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                stop = choice.finish_reason or stop
                delta = choice.delta.content or ""
                if delta:
                    parts.append(delta)
                    if on_token:
                        on_token(delta)
        except openai.APIStatusError as exc:
            raise LLMError(f"Open-weight API error {exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Could not reach the open-weight endpoint") from exc
        return LLMResult(text="".join(parts), call=self._call(purpose, usage, started, stop))

    async def structured(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        schema: type[T],
        purpose: str,
        tier: Tier = "fast",
    ) -> StructuredResult[T]:
        # Not every routed provider supports json_schema, so ask for JSON in the prompt,
        # use json_object mode, and validate with Pydantic. One repair retry on failure.
        instruction = (
            f"{system}\n\nRespond with ONLY a JSON object matching this JSON Schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        started = time.perf_counter()
        last_error: Exception | None = None
        for _ in range(2):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=self._messages(instruction, messages),
                    max_tokens=2048,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
            except openai.APIStatusError as exc:
                raise LLMError(f"Open-weight API error {exc.status_code}: {exc.message}") from exc
            except openai.APIConnectionError as exc:
                raise LLMError("Could not reach the open-weight endpoint") from exc
            raw = response.choices[0].message.content or ""
            match = _JSON_BLOCK.search(raw)
            try:
                data = schema.model_validate_json(match.group(0) if match else raw)
            except ValidationError as exc:
                last_error = exc
                continue
            return StructuredResult(
                data=data,
                call=self._call(
                    purpose, response.usage, started, response.choices[0].finish_reason
                ),
            )
        raise LLMError(f"Open-weight model returned invalid JSON: {last_error}")
