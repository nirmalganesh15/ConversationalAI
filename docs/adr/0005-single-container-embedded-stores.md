# ADR-0005 — One container with embedded stores for the demo deployment

**Status:** accepted for the demo; superseded by the production plan in `docs/productionization.md`.

## Decision
Ship one Docker image (Angular build served by FastAPI on :7860) using SQLite (users, sessions,
LangGraph checkpoints), embedded Chroma, JSON BM25 indexes and a JSON document store under
`DATA_DIR`. Embedding/reranker models are baked into the image; corpora are indexed in a
background thread on first boot. If HF persistent storage is mounted at `/data`, it is used.

## Why
A Hugging Face Docker Space gives one container and an optional volume. Zero external services
means reviewers can run the exact artifact with `docker compose up`.

## Consequences
+ Simple, reproducible, cheap; same image locally and on the Space.
− Single replica: in-process rate limiter and caches; Chroma/SQLite are not for concurrent writers
  at scale. All of these sit behind small classes so they can be swapped (pgvector, Postgres,
  Redis) without touching agents or the graph.
