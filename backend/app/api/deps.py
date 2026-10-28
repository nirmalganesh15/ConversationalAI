"""FastAPI dependencies: container access, authentication, authorization."""

from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.container import Container
from app.core.security.auth import Principal, decode_token
from app.core.security.policy import AgentName, can_perform, can_use_agent

_bearer = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def get_principal(
    container: ContainerDep,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    try:
        return decode_token(creds.credentials, container.settings.jwt_secret.get_secret_value())
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token") from exc


PrincipalDep = Annotated[Principal, Depends(get_principal)]


def ensure_agent_access(principal: Principal, agent: str) -> AgentName:
    if agent not in {a.value for a in AgentName}:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown agent '{agent}'")
    if not can_use_agent(principal.role, agent):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, f"Role '{principal.role}' cannot use agent '{agent}'"
        )
    return AgentName(agent)


def ensure_action(principal: Principal, action: str) -> None:
    if not can_perform(principal.role, action):
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"Role '{principal.role}' cannot {action}")
