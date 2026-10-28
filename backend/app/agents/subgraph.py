"""Per-agent LangGraph subgraph: plan -> run_tools -> retrieve -> generate.

All four agents are built by this one function from their AgentSpec + tools, which is
what keeps the domain agents thin and the shared core shared.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.agents.context import build_messages, fit_contexts
from app.agents.specs import AgentSpec
from app.agents.tools import AGENT_TOOLS, ALWAYS_RUN, Tool, ToolContext
from app.core.guardrails.checks import check_retrieval_confidence
from app.core.guardrails.models import GuardrailResult
from app.core.llm.base import ChatMessage, LLMError
from app.core.observability.tracing import node_span
from app.graph.events import StreamRedactor, emit, emit_guardrail
from app.graph.state import ChatState, GraphDeps

NOT_FOUND = (
    "I couldn't find this in the provided sources. Try rephrasing, or upload a document "
    "that covers it."
)

_MEDICAL_INSTRUCTION = (
    "\n\n# Safety note for this turn\nThe user is asking about their personal situation. "
    "Answer only with what the sources say in general terms, do not give individual "
    "recommendations, and suggest consulting a clinician."
)


class PlannedTool(BaseModel):
    name: str
    argument: str = Field(description="Tool argument, or 'all'")


class Plan(BaseModel):
    search_query: str = Field(
        description="Standalone search query for the knowledge base, with pronouns and "
        "references from the conversation resolved"
    )
    tools: list[PlannedTool] = Field(description="Tools to run before answering (may be empty)")


def _planner_prompt(spec: AgentSpec, tools: list[Tool]) -> str:
    listing = "\n".join(f"- {t.name}: {t.description}" for t in tools) or "(no tools)"
    return (
        f"You plan retrieval for the '{spec.title}' assistant.\n"
        "1. Rewrite the user's latest question as a standalone search query (resolve "
        "references using the conversation).\n"
        "2. Choose which tools (if any) would help answer it. Only use tools from this list:\n"
        f"{listing}\n"
        "Choose tools only when they clearly help; most factual questions need none."
    )


def build_agent_graph(spec: AgentSpec, deps: GraphDeps) -> Any:
    tools = {t.name: t for t in AGENT_TOOLS[spec.name.value]}
    retriever = deps.kb.retriever(spec.name.value)
    s = deps.settings

    async def plan(state: ChatState) -> dict[str, Any]:
        timings = dict(state["timings"])
        calls = list(state["llm_calls"])
        history = state.get("messages", [])
        query = state["query"]
        emit("status", node="plan")
        planned: list[dict[str, str]] = [
            {"name": n, "argument": "all"} for n in ALWAYS_RUN.get(spec.name.value, [])
        ]
        with node_span("plan", timings, agent=spec.name.value):
            search_query = query
            if tools or history:
                llm = deps.providers.get(state["provider"], state["model"])
                context = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])
                try:
                    result = await llm.structured(
                        system=_planner_prompt(spec, list(tools.values())),
                        messages=[
                            ChatMessage(
                                role="user",
                                content=f"CONVERSATION:\n{context or '(none)'}\n\n"
                                f"LATEST QUESTION:\n{query}",
                            )
                        ],
                        schema=Plan,
                        purpose="planner",
                    )
                    calls.append(result.call.model_dump())
                    rewrite = result.data.search_query.strip()
                    # Guard against degenerate rewrites (empty, or the prompt echoed back).
                    if rewrite and len(rewrite) <= max(300, 2 * len(query)):
                        search_query = rewrite
                    for t in result.data.tools:
                        if t.name in tools and t.name not in {p["name"] for p in planned}:
                            planned.append({"name": t.name, "argument": t.argument})
                except LLMError:
                    pass  # planning is an optimisation; fall back to the raw question
        emit("plan", search_query=search_query, tools=planned)
        return {
            "search_query": search_query,
            "planned_tools": planned,
            "timings": timings,
            "llm_calls": calls,
        }

    async def run_tools(state: ChatState) -> dict[str, Any]:
        timings = dict(state["timings"])
        calls = list(state["llm_calls"])
        outputs: list[dict[str, str]] = []
        if not state["planned_tools"]:
            return {}
        emit("status", node="tools")
        ctx = ToolContext(
            kb=deps.kb,
            owners=deps.kb.visible_owners(state["user_id"]),
            llm=deps.providers.get(state["provider"], state["model"]),
        )
        with node_span("tools", timings, agent=spec.name.value):
            for planned in state["planned_tools"][:4]:
                tool = tools[planned["name"]]
                try:
                    out = await tool.run(planned["argument"], ctx)
                except LLMError as exc:
                    outputs.append(
                        {
                            "name": tool.name,
                            "argument": planned["argument"],
                            "text": f"Tool failed: {exc}",
                        }
                    )
                    continue
                calls.extend(c.model_dump() for c in out.calls)
                outputs.append({"name": out.name, "argument": out.argument, "text": out.text})
                emit("tool", name=out.name, argument=out.argument, output=out.text)
        return {"tool_outputs": outputs, "timings": timings, "llm_calls": calls}

    async def retrieve(state: ChatState) -> dict[str, Any]:
        timings = dict(state["timings"])
        guardrails = list(state["guardrails"])
        emit("status", node="retrieve")
        with node_span("retrieve", timings, agent=spec.name.value):
            results = retriever.search(
                state["search_query"] or state["query"],
                owners=deps.kb.visible_owners(state["user_id"]),
            )
        contexts = fit_contexts(
            [r.model_dump() | {"confidence": r.confidence} for r in results],
            s.context_token_budget,
        )
        emit("sources", items=contexts)

        if results and results[0].rerank_score is None:
            check = GuardrailResult(
                name="retrieval_confidence",
                stage="retrieval",
                action="pass",
                detail="Reranker disabled - no confidence gate",
            )
        else:
            top = results[0].rerank_score if results else None
            threshold = (
                s.min_rerank_score if s.min_rerank_score is not None else spec.min_confidence
            )
            check = check_retrieval_confidence(top, threshold)
        if check.blocked and any(t["text"] for t in state["tool_outputs"]):
            # Tools (e.g. gap analysis) already produced grounded data - answer from that.
            check = check.model_copy(
                update={"action": "flag", "detail": f"{check.detail}; answering from tool results"}
            )
        guardrails.append(emit_guardrail(check))
        return {
            "contexts": contexts if not check.blocked else [],
            "guardrails": guardrails,
            "timings": timings,
            "blocked": check.blocked,
        }

    async def generate(state: ChatState) -> dict[str, Any]:
        timings = dict(state["timings"])
        calls = list(state["llm_calls"])
        emit("status", node="generate")
        if state["blocked"]:
            emit("token", text=NOT_FOUND)
            return {"answer": NOT_FOUND}
        llm = deps.providers.get(state["provider"], state["model"])
        system = spec.system_prompt
        if "medical_disclaimer" in state["flags"]:
            system += _MEDICAL_INSTRUCTION
        messages = build_messages(
            question=state["query"],
            history=state.get("messages", []),
            summary=state.get("summary", ""),
            contexts=state["contexts"],
            tool_outputs=state["tool_outputs"],
            history_turns=s.history_turns_verbatim,
        )
        redactor = StreamRedactor()
        with node_span("generate", timings, agent=spec.name.value):
            try:
                result = await llm.complete(
                    system=system, messages=messages, purpose="answer", on_token=redactor.feed
                )
            except LLMError as exc:
                message = f"⚠️ The model call failed: {exc}"
                emit("token", text=message)
                return {"answer": message, "timings": timings, "blocked": True}
            redactor.flush()
        calls.append(result.call.model_dump())
        return {"answer": result.text, "timings": timings, "llm_calls": calls}

    graph = StateGraph(ChatState)
    graph.add_node("plan", plan)
    graph.add_node("run_tools", run_tools)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "run_tools")
    graph.add_edge("run_tools", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
