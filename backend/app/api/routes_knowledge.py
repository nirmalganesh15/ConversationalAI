"""Knowledge-base endpoints: list sources, upload files, ingest a GitHub repo, delete."""

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.api.deps import ContainerDep, PrincipalDep, ensure_action, ensure_agent_access
from app.core.ingestion.loaders import UnsupportedFileError
from app.core.ingestion.pipeline import SYSTEM_OWNER, IngestReport
from app.core.security.policy import Role

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class GithubRequest(BaseModel):
    url: str


@router.get("/{agent}/sources")
def list_sources(
    agent: str, principal: PrincipalDep, container: ContainerDep
) -> list[dict[str, Any]]:
    ensure_agent_access(principal, agent)
    owners = container.kb.visible_owners(principal.user_id)
    rows = container.kb.retriever(agent).sources(owners)
    for row in rows:
        row["shared"] = row["owner"] == SYSTEM_OWNER
        row["deletable"] = row["owner"] == principal.user_id or principal.role == Role.ADMIN
        row.pop("owner", None)  # never expose other users' ids
    return rows


@router.post("/{agent}/upload", response_model=list[IngestReport])
async def upload(
    agent: str,
    principal: PrincipalDep,
    container: ContainerDep,
    files: Annotated[list[UploadFile], File()],
    doc_role: Annotated[str | None, Form()] = None,
) -> list[IngestReport]:
    ensure_agent_access(principal, agent)
    if agent == "career" and doc_role not in {"resume", "jd"}:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "doc_role must be resume or jd")
    reports: list[IngestReport] = []
    for upload_file in files:
        data = await upload_file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File exceeds 25 MB")
        name = (upload_file.filename or "upload").replace("\\", "/").split("/")[-1]
        extra = {"doc_role": doc_role} if doc_role else None
        try:
            # Embedding is CPU-bound; keep the event loop free for other requests.
            report = await asyncio.to_thread(
                container.kb.ingest_file, agent, name, data, principal.user_id, extra
            )
        except UnsupportedFileError as exc:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
        reports.append(report)
    return reports


@router.post("/code/github", response_model=IngestReport)
async def ingest_github(
    body: GithubRequest, principal: PrincipalDep, container: ContainerDep
) -> IngestReport:
    ensure_agent_access(principal, "code")
    ensure_action(principal, "ingest_github")
    try:
        return await container.kb.ingest_github(body.url, principal.user_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


@router.delete("/{agent}/sources", status_code=status.HTTP_204_NO_CONTENT)
def delete_source(
    agent: str, source: str, principal: PrincipalDep, container: ContainerDep
) -> None:
    ensure_agent_access(principal, agent)
    removed = container.kb.delete_source(agent, source, principal.user_id)
    if not removed and principal.role == Role.ADMIN:
        removed = container.kb.delete_source(agent, source, SYSTEM_OWNER)
    if not removed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Source not found or not yours")
