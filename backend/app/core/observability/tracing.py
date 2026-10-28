"""OpenTelemetry tracing. Exporter is chosen by env so prod can point at any OTLP backend."""

import time
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from app.core.observability.metrics import LATENCY

_tracer = trace.get_tracer("atlas")


def configure_tracing(service_name: str, exporter: str) -> None:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        # Imported lazily: the OTLP exporter is an optional dependency for prod.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


@contextmanager
def node_span(node: str, timings: dict[str, float], **attributes: str) -> Iterator[None]:
    """Trace a graph node and record its wall time into the per-turn timings dict."""
    start = time.perf_counter()
    with _tracer.start_as_current_span(f"node.{node}", attributes=attributes):
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            timings[node] = round(elapsed * 1000, 1)
            LATENCY.labels(node=node).observe(elapsed)
