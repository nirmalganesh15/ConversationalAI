# Developer entry points. Run from the repo root.
.PHONY: setup dev-api dev-web test lint typecheck security check eval-retrieval eval reindex-code build docker deploy-hf

setup:            ## install backend + frontend dependencies
	cd backend && uv sync --all-extras
	cd frontend && npm ci

dev-api:          ## FastAPI with reload on :8000
	cd backend && uv run python -m uvicorn app.main:app --reload --port 8000

dev-web:          ## Angular dev server on :4200 (proxies /api to :8000)
	cd frontend && npx ng serve

test:             ## backend + frontend unit/integration tests
	cd backend && uv run pytest -q
	cd frontend && npx ng test --watch=false

lint:
	cd backend && uv run ruff check . && uv run ruff format --check .

typecheck:
	cd backend && uv run mypy app

security:         ## SAST + dependency audit
	cd backend && uv run bandit -q -r app -c pyproject.toml
	cd backend && uvx pip-audit

check: lint typecheck test security

eval-retrieval:   ## retrieval ablation (no LLM, free)
	cd backend && uv run python -m evals.run retrieval

eval:             ## end-to-end eval with quality gates (uses the configured LLM)
	cd backend && uv run python -m evals.run e2e --agent all --provider claude

reindex-code:     ## re-index this repository into the Code agent after changes
	cd backend && uv run python -m app.seed --agent code

build:            ## build SPA into backend/static (single-process local run)
	cd frontend && npx ng build && rm -rf ../backend/static && cp -r dist/browser ../backend/static

docker:
	docker compose up --build

deploy-hf:        ## push to the Hugging Face Space (needs HF_TOKEN and HF_SPACE=user/space)
	cd backend && uv run --with huggingface_hub python scripts/deploy_hf.py
