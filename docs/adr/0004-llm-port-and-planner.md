# ADR-0004 — LLM provider port and a structured-output planner

**Status:** accepted · **Context:** the UI must switch between Claude and an open-weight model per
message; open-weight endpoints differ in tool-calling and JSON-schema support.

## Decision
- One `LLMProvider` protocol: `complete(..., on_token)` for streamed text and
  `structured(..., schema)` for Pydantic-validated JSON. Adapters: Anthropic SDK; any
  OpenAI-compatible endpoint (HF Inference Providers by default; Ollama/Groq/vLLM by env).
- Two tiers: `main` (the answer) and `fast` (planner, classifier, judge, summaries).
- Tool selection uses a **structured-output planner** (one fast call returns the rewritten query
  and tool calls) instead of native tool calling. Tools are allow-listed per agent and executed by
  the graph.
- A `FakeProvider` implements the same port: tests are offline and deterministic, and the app has
  an honest offline demo mode.

## Consequences
+ Identical behaviour across providers; the planner is cheap and fully testable.
+ Cost/latency/tokens are recorded uniformly for the UI and Prometheus.
− No open-ended multi-hop tool loops. For Claude, a native tool loop would be a natural upgrade,
  keeping the planner as the portable path.
