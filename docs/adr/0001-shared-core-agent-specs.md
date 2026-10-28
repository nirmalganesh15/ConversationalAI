# ADR-0001 — One shared core, agents as declarative specs

**Status:** accepted · **Context:** the assignment offers four use cases; the brief values a solid
core over breadth. Building four apps would quadruple ingestion, retrieval, guardrails and evals.

## Decision
All four use cases are RAG over a document type with a domain-specific twist. The twist is
captured in an `AgentSpec` (analyzer, chunker, prompt, tools, confidence threshold, accepted file
types, example questions). One builder turns any spec into the same LangGraph subgraph
(`plan → run_tools → retrieve → generate`). The supervisor routes to it after guardrails and RBAC.

## Consequences
+ Guardrails, evals, observability, memory and auth are written once and apply to every agent.
+ A new agent is ~1 spec + 1 prompt + 1 eval set (`/new-agent` command).
− Agents that need a radically different control flow (e.g. long multi-hop research) would need a
  custom subgraph; the supervisor already supports any compiled subgraph per agent.
− Domain behaviour that does not fit "retrieve then answer" moves into tools (gap analysis,
  decision extraction), which is a deliberate constraint.
