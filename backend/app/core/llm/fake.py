"""Deterministic provider for tests and for running the app without API keys.

Without a canned answer it behaves as an *extractive* model: it quotes the opening
sentences of the top-ranked passage and cites it, so the offline demo still shows real
retrieval, citations, guardrails and the inspector end-to-end.
"""

import re
from collections.abc import Callable

from app.core.llm.base import (
    ChatMessage,
    LLMCall,
    LLMResult,
    StructuredResult,
    T,
    Tier,
    Usage,
)


class FakeProvider:
    name = "fake"

    def __init__(
        self,
        answer: str | None = None,
        structured_factory: Callable[[type], object] | None = None,
    ) -> None:
        self.answer = answer
        self.structured_factory = structured_factory
        self.calls: list[str] = []

    def model_for(self, tier: Tier) -> str:
        return f"fake-{tier}"

    def with_main_model(self, model: str) -> "FakeProvider":
        return self

    def _call(self, purpose: str, tier: Tier) -> LLMCall:
        self.calls.append(purpose)
        return LLMCall(
            purpose=purpose,
            provider=self.name,
            model=self.model_for(tier),
            usage=Usage(input_tokens=100, output_tokens=20),
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
        text = self.answer if self.answer is not None else _extractive(messages)
        if on_token:
            for word in text.split(" "):
                on_token(word + " ")
        return LLMResult(text=text, call=self._call(purpose, tier))

    async def structured(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        schema: type[T],
        purpose: str,
        tier: Tier = "fast",
    ) -> StructuredResult[T]:
        if self.structured_factory:
            data = self.structured_factory(schema)
        else:
            # Build a schema instance from defaults; tests override where it matters.
            data = schema.model_validate(_defaults_for(schema, messages))
        return StructuredResult(data=data, call=self._call(purpose, tier))  # type: ignore[arg-type]


def _defaults_for(schema: type, messages: list[ChatMessage]) -> dict[str, object]:
    question = messages[-1].content if messages else ""
    question = question.rsplit("LATEST QUESTION:\n", 1)[-1]  # planner prompt -> the question
    values: dict[str, object] = {}
    for name, field in schema.model_fields.items():  # type: ignore[attr-defined]
        ann = str(field.annotation)
        if not field.is_required():
            continue
        if "float" in ann:
            values[name] = 1.0
        elif "bool" in ann:
            values[name] = False
        elif "int" in ann:
            values[name] = 1
        elif "list" in ann:
            values[name] = []
        else:
            values[name] = question
    return values


def _extractive(messages: list[ChatMessage]) -> str:
    prompt = messages[-1].content if messages else ""
    match = _TOP_PASSAGE.search(prompt)
    if not match:
        return "I couldn't find this in the provided sources."
    body = re.sub(r"^#.*\n", "", match["text"]).strip()
    sentences = re.split(r"(?<=[.!?])\s+", body)
    excerpt = " ".join(sentences[:3])[:700]
    return (
        "**Offline demo mode** (no LLM key configured) - extractive answer from the top "
        f"passage, *{match['label']}*:\n\n> {excerpt} [1]"
    )


_TOP_PASSAGE = re.compile(
    r"\[1\] \((?P<label>[^\n]*)\)\n(?P<text>.*?)(?:\n\n\[2\] |\n\nQUESTION:)", re.S
)
