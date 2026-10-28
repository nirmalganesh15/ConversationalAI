"""Single source of truth for authorization (least privilege).

Every enforcement point - API dependency, graph `authorize` node, ingestion routes,
MCP server - calls into this module, so the RBAC matrix lives in exactly one place.
"""

from enum import StrEnum


class Role(StrEnum):
    ADMIN = "admin"
    STAFF = "staff"


class AgentName(StrEnum):
    DOCS = "docs"
    CODE = "code"
    MEETING = "meeting"
    CAREER = "career"


_AGENT_ACCESS: dict[Role, frozenset[AgentName]] = {
    Role.ADMIN: frozenset(AgentName),
    Role.STAFF: frozenset({AgentName.DOCS, AgentName.CODE}),
}

# Admin-only capabilities beyond agents.
_ADMIN_ONLY_ACTIONS = frozenset({"view_eval_reports", "run_offline_evals", "ingest_github"})


def allowed_agents(role: Role | str) -> list[AgentName]:
    return sorted(_AGENT_ACCESS.get(Role(role), frozenset()))


def can_use_agent(role: Role | str, agent: AgentName | str) -> bool:
    try:
        return AgentName(agent) in _AGENT_ACCESS.get(Role(role), frozenset())
    except ValueError:
        return False  # unknown role or agent -> deny by default


def can_perform(role: Role | str, action: str) -> bool:
    if action in _ADMIN_ONLY_ACTIONS:
        return Role(role) is Role.ADMIN
    return True
