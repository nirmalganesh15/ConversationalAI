"""MCP server exposing Atlas retrieval to MCP clients (Claude Code, Claude Desktop, ...).

    uv run --extra mcp python -m app.mcp_server            # stdio transport

Register with Claude Code:
    claude mcp add atlas -- uv --directory backend run --extra mcp python -m app.mcp_server

The server acts with a fixed role (MCP_ROLE, default 'staff') and goes through the same
policy module as the web API, so an MCP client can never read agents its role excludes.
"""

import os
from typing import Any

from fastmcp import FastMCP

from app.agents.context import source_label
from app.agents.specs import SPECS
from app.container import Container, build_container
from app.core.config import get_settings
from app.core.ingestion.pipeline import SYSTEM_OWNER, KnowledgeBase
from app.core.security.policy import Role, allowed_agents, can_use_agent

ROLE = Role(os.environ.get("MCP_ROLE", Role.STAFF.value))
mcp = FastMCP("atlas")
_container: Container | None = None


def _kb() -> KnowledgeBase:
    global _container  # lazy: models load only when a tool is first used
    if _container is None:
        _container = build_container(get_settings())
    return _container.kb


@mcp.tool
def list_agents() -> list[dict[str, str]]:
    """List the knowledge bases (agents) this MCP connection may search."""
    return [
        {"name": a.value, "title": SPECS[a].title, "description": SPECS[a].description}
        for a in allowed_agents(ROLE)
    ]


@mcp.tool
def search_knowledge(agent: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Hybrid search (stemmed BM25 + embeddings + rerank) over one agent's knowledge base.

    Returns passages with their source label and relevance score, best first.
    """
    if not can_use_agent(ROLE, agent):
        raise ValueError(f"Role '{ROLE}' may not search agent '{agent}'")
    results = _kb().retriever(agent).search(query, {SYSTEM_OWNER}, top_k=min(top_k, 10))
    return [
        {
            "source": source_label(r.chunk.metadata),
            "score": round(r.confidence, 3),
            "text": r.chunk.text,
        }
        for r in results
    ]


if __name__ == "__main__":
    mcp.run()
