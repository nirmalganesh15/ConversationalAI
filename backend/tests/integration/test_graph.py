"""Graph-level behaviour that does not need the HTTP layer."""

from langgraph.checkpoint.memory import InMemorySaver

from app.agents.subgraph import Plan, PlannedTool
from app.container import Container
from app.core.llm.fake import FakeProvider
from app.core.llm.registry import ProviderInfo, ProviderRegistry
from app.graph.state import GraphDeps, new_turn
from app.graph.supervisor import build_graph


def _graph(container: Container, llm: FakeProvider):  # type: ignore[no-untyped-def]
    info = ProviderInfo(key="fake", label="F", models=["f"], default_model="f", available=True)
    registry = ProviderRegistry({"fake": llm}, [info])
    deps = GraphDeps(kb=container.kb, providers=registry, settings=container.settings)
    return build_graph(deps, InMemorySaver())


async def _run(graph, question: str, agent: str = "docs"):  # type: ignore[no-untyped-def]
    return await graph.ainvoke(
        new_turn(
            user_id="u", role="admin", agent=agent, provider="fake", model="f", question=question
        ),
        {"configurable": {"thread_id": "u:t"}},
    )


async def test_degenerate_planner_rewrite_falls_back_to_question(container: Container) -> None:
    """Regression: a planner that echoes its prompt must not poison retrieval."""
    llm = FakeProvider(
        answer="Dose is 2550 mg [1].",
        structured_factory=lambda schema: Plan(
            search_query="CONVERSATION: " + "x " * 400, tools=[]
        ),
    )
    graph = _graph(container, llm)
    await _run(graph, "first question about metformin")  # creates history -> planner runs
    state = await _run(graph, "What is the maximum dose of metformin?")
    assert state["search_query"] == "What is the maximum dose of metformin?"


async def test_planner_cannot_invoke_tools_outside_the_agent(container: Container) -> None:
    """Least privilege for tools: unknown / other-agent tool names are dropped."""
    llm = FakeProvider(
        answer="ok [1]",
        structured_factory=lambda schema: (
            Plan(
                search_query="decisions",
                tools=[
                    PlannedTool(name="gap_analysis", argument="all"),
                    PlannedTool(name="extract_decisions", argument="all"),
                ],
            )
            if schema is Plan
            else schema(items=[])
        ),  # the extraction tool's own structured call
    )
    graph = _graph(container, llm)
    state = await _run(graph, "What was decided?", agent="meeting")
    names = [t["name"] for t in state["planned_tools"]]
    assert "gap_analysis" not in names  # career tool, not allowed for the meeting agent
    assert "extract_decisions" in names


async def test_memory_compacts_old_turns_into_summary(container: Container) -> None:
    container.settings.history_turns_verbatim = 1
    graph = _graph(container, FakeProvider(answer="Answer [1]."))
    for q in ("metformin dose?", "lisinopril contraindications?", "metformin eGFR?"):
        state = await _run(graph, q)
    assert len(state["messages"]) == 2  # only the last turn verbatim
    assert state["summary"]  # older turns rolled into the running summary
    assert len(state["transcript"]) == 3  # the UI still sees full history
