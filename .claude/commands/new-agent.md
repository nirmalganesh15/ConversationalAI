---
description: Scaffold a new domain agent on the shared RAG core (spec, prompt, tools, eval set, tests)
argument-hint: <agent-name> "<one-line purpose>"
---

Add a new Atlas agent named `$1` whose purpose is: $2

Follow CLAUDE.md. Steps:
1. Add `$1` to `AgentName` in `backend/app/core/security/policy.py` and decide which roles may
   use it (least privilege: admin only unless told otherwise).
2. Add an `AgentSpec` in `backend/app/agents/specs.py`: pick the analyzer (stemming vs code) and
   the chunker that matches the document shape; justify `min_confidence`.
3. Write `backend/app/agents/prompts/$1.md` (role + domain guidance only; grounding rules come
   from `_shared.md`).
4. If the agent needs whole-document reasoning, add tools to `AGENT_TOOLS["$1"]`.
5. Add `backend/evals/datasets/$1.yaml` with at least 6 verified questions + 1 out-of-scope.
6. Tests: RBAC for the new agent, chunker behaviour, one integration chat test.
7. Run `make check` and `uv run python -m evals.run retrieval --agent $1`; report the numbers.
