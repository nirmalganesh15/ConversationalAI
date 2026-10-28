"""Prometheus metrics exposed on /metrics."""

from prometheus_client import Counter, Histogram

REQUESTS = Counter("atlas_chat_requests_total", "Chat requests", ["agent", "provider", "outcome"])
LATENCY = Histogram(
    "atlas_node_latency_seconds",
    "Latency per graph node",
    ["node"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
TOKENS = Counter("atlas_llm_tokens_total", "LLM tokens", ["provider", "model", "kind"])
COST = Counter("atlas_llm_cost_usd_total", "Estimated LLM spend (USD)", ["provider", "model"])
GUARDRAIL_TRIPS = Counter(
    "atlas_guardrail_trips_total", "Guardrail flags/blocks", ["name", "action"]
)
