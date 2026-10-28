"""FastAPI entrypoint. Serves the API and (in the container) the built Angular SPA."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api import routes_auth, routes_chat, routes_evals, routes_knowledge
from app.container import Container, build_container
from app.core.config import Settings, get_settings
from app.core.observability.logging import configure_logging, get_logger
from app.core.observability.tracing import configure_tracing
from app.graph.state import GraphDeps
from app.graph.supervisor import build_graph
from app.seed import seed

log = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json=settings.env != "dev")
    configure_tracing(settings.app_name, settings.otel_exporter)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        c = container or build_container(settings)
        async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_path)) as saver:
            c.graph = build_graph(
                GraphDeps(kb=c.kb, providers=c.providers, settings=settings), saver
            )
            app.state.container = c
            task = None
            if settings.seed_on_startup:
                # Seeding loads embedding models; run it off the event loop so the port
                # opens immediately (HF Spaces health-checks the port on boot).
                task = asyncio.create_task(asyncio.to_thread(seed, c, REPO_ROOT))
            log.info("startup", providers=[p.key for p in c.providers.infos if p.available])
            yield
            if task and not task.done():
                task.cancel()

    app = FastAPI(title="Atlas API", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:16])
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["x-request-id"] = request_id
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["referrer-policy"] = "no-referrer"
        return response

    for module in (routes_auth, routes_chat, routes_knowledge, routes_evals):
        app.include_router(module.router)

    @app.get("/health", tags=["ops"])
    def health(request: Request) -> dict[str, object]:
        c: Container | None = getattr(request.app.state, "container", None)
        return {"status": "ok", "seed": c.status["seed"] if c else "starting"}

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    _mount_spa(app, settings.static_dir)
    return app


def _mount_spa(app: FastAPI, static_dir: Path) -> None:
    """Serve the Angular build; unknown non-API paths fall back to index.html."""
    index = static_dir / "index.html"
    if not index.exists():
        return

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        candidate = (static_dir / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(static_dir.resolve()):
            return FileResponse(candidate)
        return FileResponse(index)


app = create_app()
