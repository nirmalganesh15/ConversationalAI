from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.agents.specs import SPECS
from app.api.deps import ContainerDep, PrincipalDep
from app.core.security.auth import issue_token
from app.core.security.policy import allowed_agents, can_perform

router = APIRouter(prefix="/api", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class UserInfo(BaseModel):
    username: str
    role: str
    allowed_agents: list[str]
    is_admin: bool


class LoginResponse(BaseModel):
    token: str
    user: UserInfo


def _user_info(username: str, role: str) -> UserInfo:
    return UserInfo(
        username=username,
        role=role,
        allowed_agents=[a.value for a in allowed_agents(role)],
        is_admin=can_perform(role, "view_eval_reports"),
    )


@router.post("/auth/login", response_model=LoginResponse)
def login(body: LoginRequest, container: ContainerDep) -> LoginResponse:
    principal = container.db.authenticate(body.username, body.password)
    if principal is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password")
    token = issue_token(
        principal,
        container.settings.jwt_secret.get_secret_value(),
        container.settings.jwt_ttl_minutes,
    )
    return LoginResponse(token=token, user=_user_info(principal.username, principal.role))


@router.get("/auth/me", response_model=UserInfo)
def me(principal: PrincipalDep) -> UserInfo:
    return _user_info(principal.username, principal.role)


@router.get("/config")
def app_config(principal: PrincipalDep, container: ContainerDep) -> dict[str, object]:
    """Everything the UI needs to render selectors - already filtered by role."""
    permitted = set(allowed_agents(principal.role))
    return {
        "agents": [
            {
                "name": spec.name.value,
                "title": spec.title,
                "description": spec.description,
                "icon": spec.icon,
                "accepts": list(spec.accepts),
                "examples": list(spec.example_questions),
            }
            for name, spec in SPECS.items()
            if name in permitted
        ],
        "providers": [p.model_dump() for p in container.providers.infos],
        "retrieval": {
            "embedding_model": container.settings.embedding_model,
            "reranker_model": container.settings.reranker_model
            if container.settings.reranker_enabled
            else None,
            "top_k": container.settings.context_top_k,
        },
        "status": container.status,
    }
