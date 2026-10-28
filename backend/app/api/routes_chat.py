"""Chat (SSE streaming) and session endpoints."""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.api.deps import ContainerDep, PrincipalDep, ensure_agent_access
from app.container import Container
from app.core.db import ChatSession
from app.core.observability.logging import get_logger
from app.core.observability.metrics import REQUESTS
from app.core.security.auth import Principal
from app.graph.state import new_turn

router = APIRouter(prefix="/api", tags=["chat"])
log = get_logger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    agent: str
    provider: str = "claude"
    model: str | None = None
    session_id: str | None = None


def _thread(principal: Principal, session_id: str) -> dict[str, Any]:
    # Thread ids are namespaced by user, so one user can never load another's memory.
    return {"configurable": {"thread_id": f"{principal.user_id}:{session_id}"}}


async def _stream(
    container: Container, principal: Principal, body: ChatRequest, session: ChatSession
) -> AsyncIterator[dict[str, str]]:
    config = _thread(principal, session.id)
    provider = container.providers.get(body.provider, body.model)
    inputs = new_turn(
        user_id=principal.user_id,
        role=principal.role.value,
        agent=body.agent,
        provider=body.provider,
        model=body.model or provider.model_for("main"),
        question=body.question,
    )
    yield {"event": "session", "data": json.dumps(session.model_dump())}
    outcome = "ok"
    try:
        async for _namespace, event in container.graph.astream(
            inputs, config, stream_mode="custom", subgraphs=True
        ):
            yield {"event": event["type"], "data": json.dumps(event, default=str)}
        final = (await container.graph.aget_state(config)).values
        if final.get("blocked"):
            outcome = "blocked"
        yield {
            "event": "done",
            "data": json.dumps({"answer": final.get("answer", ""), "session_id": session.id}),
        }
    except Exception as exc:  # surface to the UI instead of a silently dropped stream
        outcome = "error"
        log.exception("chat_failed", session_id=session.id)
        yield {"event": "error", "data": json.dumps({"message": str(exc)})}
    finally:
        container.db.touch_session(session.id)
        REQUESTS.labels(body.agent, body.provider, outcome).inc()


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest, principal: PrincipalDep, container: ContainerDep
) -> EventSourceResponse:
    ensure_agent_access(principal, body.agent)
    if not container.rate_limiter.allow(principal.user_id):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded")
    session = (
        container.db.get_session(principal.user_id, body.session_id) if body.session_id else None
    )
    if body.session_id and session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    if session is None:
        session = container.db.create_session(principal.user_id, body.agent, body.question)
    return EventSourceResponse(_stream(container, principal, body, session), ping=15)


@router.get("/sessions", response_model=list[ChatSession])
def list_sessions(principal: PrincipalDep, container: ContainerDep) -> list[ChatSession]:
    return container.db.list_sessions(principal.user_id)


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str, principal: PrincipalDep, container: ContainerDep
) -> dict[str, Any]:
    session = container.db.get_session(principal.user_id, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    state = await container.graph.aget_state(_thread(principal, session_id))
    values = state.values or {}
    return {
        "session": session.model_dump(),
        "turns": values.get("transcript", []),
        "summary": values.get("summary", ""),
    }


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, principal: PrincipalDep, container: ContainerDep) -> None:
    if not container.db.delete_session(principal.user_id, session_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    await container.graph.checkpointer.adelete_thread(f"{principal.user_id}:{session_id}")
