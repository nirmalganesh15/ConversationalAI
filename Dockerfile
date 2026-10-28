# syntax=docker/dockerfile:1.7
# Single image: Angular SPA + FastAPI API on port 7860 (Hugging Face Docker Space convention).

# ---- 1. build the Angular SPA -----------------------------------------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json frontend/.npmrc ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npx ng build --configuration production

# ---- 2. python runtime -------------------------------------------------------------------
FROM python:3.12-slim AS runtime
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /usr/local/bin/uv

# HF Spaces run containers as uid 1000; own everything we need to write.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    APP=/home/user/app \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/app/backend/.venv/bin:$PATH
WORKDIR $APP/backend

# Dependencies first for layer caching (CPU-only torch via the index in pyproject).
COPY --chown=user backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Bake the embedding + reranker models into the image: no download on cold start.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}'); CrossEncoder('${RERANKER_MODEL}')"

# Application code, sample corpora, and the sources the Code agent indexes about itself.
COPY --chown=user backend/ ./
COPY --chown=user samples/ $APP/samples/
COPY --chown=user docs/ $APP/docs/
COPY --chown=user README.md Dockerfile CLAUDE.md $APP/
COPY --chown=user frontend/src $APP/frontend/src
COPY --chown=user frontend/package.json $APP/frontend/package.json
COPY --chown=user .github $APP/.github
COPY --chown=user --from=web /web/dist/browser ./static

ENV ENV=prod \
    DATA_DIR=/home/user/data \
    SAMPLES_DIR=/home/user/app/samples \
    STATIC_DIR=/home/user/app/backend/static \
    PORT=7860 \
    HF_HUB_OFFLINE=1
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# If an HF persistent volume is mounted at /data, use it so indexes and chats survive restarts.
CMD ["sh", "-c", "if [ -w /data ]; then export DATA_DIR=/data; fi; exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
