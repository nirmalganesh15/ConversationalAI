"""Custom stream events emitted from graph nodes and forwarded to the browser as SSE."""

from typing import Any

from langgraph.config import get_stream_writer

from app.core.guardrails.checks import redact_pii
from app.core.guardrails.models import GuardrailResult
from app.core.observability.metrics import GUARDRAIL_TRIPS


def emit(event_type: str, **payload: Any) -> None:
    get_stream_writer()({"type": event_type, **payload})


def emit_guardrail(result: GuardrailResult) -> dict[str, Any]:
    if result.action != "pass":
        GUARDRAIL_TRIPS.labels(name=result.name, action=result.action).inc()
    item = result.model_dump()
    emit("guardrail", item=item)
    return item


class StreamRedactor:
    """Applies PII redaction to a token stream.

    Tokens are buffered up to a whitespace boundary so a pattern (e.g. an email) is
    never emitted half-way. The final stored answer is redacted again as a whole.
    """

    def __init__(self, min_flush: int = 60) -> None:
        self._buffer = ""
        self._min_flush = min_flush

    def feed(self, text: str) -> None:
        self._buffer += text
        if "\n" in self._buffer:
            cut = self._buffer.rfind("\n") + 1
        elif len(self._buffer) >= self._min_flush:
            cut = self._buffer.rfind(" ") + 1
        else:
            return
        if cut <= 0:
            return
        head, self._buffer = self._buffer[:cut], self._buffer[cut:]
        emit("token", text=redact_pii(head)[0])

    def flush(self) -> None:
        if self._buffer:
            emit("token", text=redact_pii(self._buffer)[0])
            self._buffer = ""
