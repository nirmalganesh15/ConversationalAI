import pytest

from app.core.security.policy import AgentName, Role, allowed_agents, can_perform, can_use_agent


def test_admin_can_use_every_agent() -> None:
    assert set(allowed_agents(Role.ADMIN)) == set(AgentName)


def test_staff_limited_to_docs_and_code() -> None:
    assert allowed_agents(Role.STAFF) == [AgentName.CODE, AgentName.DOCS]


@pytest.mark.parametrize("agent", ["meeting", "career"])
def test_staff_denied_restricted_agents(agent: str) -> None:
    assert not can_use_agent("staff", agent)


@pytest.mark.parametrize("role, agent", [("guest", "docs"), ("admin", "unknown")])
def test_unknown_role_or_agent_is_denied(role: str, agent: str) -> None:
    assert not can_use_agent(role, agent)


def test_admin_only_actions() -> None:
    assert can_perform("admin", "ingest_github")
    assert not can_perform("staff", "ingest_github")
    assert not can_perform("staff", "view_eval_reports")
