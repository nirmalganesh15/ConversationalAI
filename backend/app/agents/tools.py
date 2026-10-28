"""Agent tools. The planner chooses them; the agent subgraph executes them.

Deterministic tools (endpoint listing, symbol lookup, job listing) are plain code;
extraction tools (decisions, action items, gap analysis) use structured LLM output and
are cached per (document, model) so repeated questions cost nothing.
"""

import re
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, Field

from app.core.ingestion.pipeline import KnowledgeBase, StoredDocument
from app.core.llm.base import ChatMessage, LLMCall, LLMProvider


@dataclass
class ToolContext:
    kb: KnowledgeBase
    owners: set[str]
    llm: LLMProvider


@dataclass
class ToolOutput:
    name: str
    argument: str
    text: str
    calls: list[LLMCall] = field(default_factory=list)


class Tool(Protocol):
    name: str
    description: str

    async def run(self, argument: str, ctx: ToolContext) -> ToolOutput: ...


_CACHE: dict[tuple[str, ...], BaseModel] = {}


def _expect[M: BaseModel](value: BaseModel, kind: type[M]) -> M:
    if not isinstance(value, kind):
        raise TypeError(f"expected {kind.__name__}, got {type(value).__name__}")
    return value


def _pick(docs: list[StoredDocument], ref: str, label: str) -> list[StoredDocument]:
    """Resolve 'all', '#2', '2', or a title fragment to documents."""
    ref = (ref or "").strip().lower()
    if not ref or ref == "all":
        return docs
    number = re.search(r"#?\s*(\d+)", ref)
    if number and ref.replace("#", "").replace(label, "").strip().isdigit():
        idx = int(number.group(1)) - 1
        return [docs[idx]] if 0 <= idx < len(docs) else []
    words = [w for w in re.findall(r"\w+", ref) if len(w) > 2 and w != label]
    scored = sorted(
        ((sum(w in (d.title + d.source).lower() for w in words), d) for d in docs),
        key=lambda pair: -pair[0],
    )
    return [scored[0][1]] if scored and scored[0][0] > 0 else []


# --- code agent ---------------------------------------------------------------------------
_ROUTE_PATTERNS = [
    # FastAPI / Flask style decorators
    re.compile(r"@\w+\.(get|post|put|patch|delete)\(\s*[\"']([^\"']*)[\"']"),
    # Express / Hono / Fastify style calls
    re.compile(r"\b(?:app|router|server)\.(get|post|put|patch|delete)\(\s*[\"'`]([^\"'`]+)"),
]
_PREFIX = re.compile(r"APIRouter\(\s*prefix\s*=\s*[\"']([^\"']+)")


class ListEndpoints:
    name = "list_endpoints"
    description = "List every HTTP API endpoint (method, path, file) in the indexed code."

    async def run(self, argument: str, ctx: ToolContext) -> ToolOutput:
        rows: set[tuple[str, str, str]] = set()
        prefixes: dict[str, str] = {}
        chunks = ctx.kb.retriever("code").scan(ctx.owners)
        for chunk in chunks:
            if m := _PREFIX.search(chunk.text):
                prefixes[str(chunk.metadata.get("path"))] = m.group(1)
        for chunk in chunks:
            path = str(chunk.metadata.get("path", chunk.metadata.get("source")))
            for pattern in _ROUTE_PATTERNS:
                for method, route in pattern.findall(chunk.text):
                    rows.add((method.upper(), prefixes.get(path, "") + route, path))
        if not rows:
            return ToolOutput(self.name, argument, "No HTTP endpoints found in the indexed code.")
        table = "\n".join(f"| {m} | `{r}` | {p} |" for m, r, p in sorted(rows, key=lambda x: x[1]))
        return ToolOutput(self.name, argument, f"| Method | Path | File |\n|---|---|---|\n{table}")


class FindSymbol:
    name = "find_symbol"
    description = "Find where a function/class/symbol is defined. Argument: the symbol name."

    async def run(self, argument: str, ctx: ToolContext) -> ToolOutput:
        needle = argument.strip().split(".")[-1].lower()
        if not needle:
            return ToolOutput(self.name, argument, "No symbol name given.")
        hits = sorted(
            {
                (
                    str(c.metadata.get("path")),
                    str(c.metadata.get("symbol")),
                    int(c.metadata.get("line_start", 0)),
                )
                for c in ctx.kb.retriever("code").scan(ctx.owners)
                if needle in str(c.metadata.get("symbol", "")).lower()
            }
        )[:15]
        if not hits:
            return ToolOutput(self.name, argument, f"No symbol matching '{argument}' found.")
        lines = [f"- `{sym}` in {path} (line {line})" for path, sym, line in hits]
        return ToolOutput(self.name, argument, "\n".join(lines))


class ReadDependencies:
    name = "read_dependencies"
    description = "Show declared dependencies (pyproject.toml, package.json, requirements)."

    _FILES = ("pyproject.toml", "package.json", "requirements.txt", "go.mod", "pom.xml")

    async def run(self, argument: str, ctx: ToolContext) -> ToolOutput:
        parts: list[str] = []
        for chunk in sorted(
            ctx.kb.retriever("code").scan(ctx.owners),
            key=lambda c: (str(c.metadata.get("path")), int(c.metadata.get("line_start", 0))),
        ):
            path = str(chunk.metadata.get("path", ""))
            if path.endswith(self._FILES) and chunk.metadata.get("symbol") != "<file>":
                parts.append(chunk.text[:2500])
        text = "\n\n".join(parts) if parts else "No dependency manifests found."
        return ToolOutput(self.name, argument, text[:8000])


# --- meeting agent -----------------------------------------------------------------------------
class ActionItem(BaseModel):
    owner: str = Field(description="Person responsible, or 'not specified'")
    task: str
    due: str = Field(description="Due date/timeframe as stated, or 'not specified'")
    timestamp: str = Field(description="Transcript timestamp where it was assigned")


class Decision(BaseModel):
    decision: str
    made_by: str = Field(description="Who proposed/confirmed it")
    timestamp: str
    rationale: str = Field(description="Short reason given, or ''")


class ActionItems(BaseModel):
    items: list[ActionItem]


class Decisions(BaseModel):
    items: list[Decision]


class ListMeetings:
    name = "list_meetings"
    description = "List available meeting transcripts."

    async def run(self, argument: str, ctx: ToolContext) -> ToolOutput:
        docs = ctx.kb.documents("meeting").list(ctx.owners)
        text = "\n".join(f"- Meeting #{i}: {d.title} ({d.source})" for i, d in enumerate(docs, 1))
        return ToolOutput(self.name, argument, text or "No meetings indexed.")


class _MeetingExtractor:
    name = ""
    description = ""
    schema: type[ActionItems] | type[Decisions]
    instruction = ""

    async def run(self, argument: str, ctx: ToolContext) -> ToolOutput:
        docs = _pick(ctx.kb.documents("meeting").list(ctx.owners), argument, "meeting")
        if not docs:
            return ToolOutput(self.name, argument, f"No meeting matches '{argument}'.")
        sections: list[str] = []
        calls: list[LLMCall] = []
        for doc in docs[:4]:
            key = (self.name, doc.owner, doc.source, ctx.llm.model_for("fast"))
            cached = _CACHE.get(key)
            if cached is None:
                result = await ctx.llm.structured(
                    system=self.instruction,
                    messages=[ChatMessage(role="user", content=doc.text[:60_000])],
                    schema=self.schema,
                    purpose=f"tool:{self.name}",
                )
                cached = _CACHE[key] = result.data
                calls.append(result.call)
            sections.append(f"### {doc.title}\n{self._render(cached)}")
        return ToolOutput(self.name, argument, "\n\n".join(sections), calls)

    def _render(self, data: BaseModel) -> str:
        raise NotImplementedError


class ExtractActionItems(_MeetingExtractor):
    name = "extract_action_items"
    description = "Extract action items (owner, task, due date). Argument: meeting # or 'all'."
    schema = ActionItems
    instruction = (
        "Extract every action item from this meeting transcript. An action item is a task "
        "someone agreed or was asked to do. Use the speaker names and timestamps exactly."
    )

    def _render(self, data: BaseModel) -> str:
        data = _expect(data, ActionItems)
        rows = [f"| {i.owner} | {i.task} | {i.due} | {i.timestamp} |" for i in data.items]
        return "| Owner | Task | Due | Time |\n|---|---|---|---|\n" + "\n".join(rows)


class ExtractDecisions(_MeetingExtractor):
    name = "extract_decisions"
    description = "Extract decisions made. Argument: meeting # or 'all'."
    schema = Decisions
    instruction = (
        "Extract every decision that was agreed in this meeting transcript (not proposals that "
        "were left open). Use the speaker names and timestamps exactly."
    )

    def _render(self, data: BaseModel) -> str:
        data = _expect(data, Decisions)
        rows = [
            f"| {d.decision} | {d.made_by} | {d.timestamp} | {d.rationale} |" for d in data.items
        ]
        return "| Decision | By | Time | Rationale |\n|---|---|---|---|\n" + "\n".join(rows)


# --- career agent --------------------------------------------------------------------------------
class SkillMatch(BaseModel):
    skill: str
    status: str = Field(description="One of: matched, partial, missing")
    evidence: str = Field(description="Resume evidence, or what is lacking")


class GapAnalysis(BaseModel):
    role: str
    fit_score: int = Field(ge=0, le=100, description="Overall fit 0-100")
    skills: list[SkillMatch]
    experience_alignment: str
    top_gaps_to_close: list[str]


def _resumes(ctx: ToolContext) -> list[StoredDocument]:
    return ctx.kb.documents("career").list(ctx.owners, doc_role="resume")


def _jobs(ctx: ToolContext) -> list[StoredDocument]:
    return ctx.kb.documents("career").list(ctx.owners, doc_role="jd")


class ListJobs:
    name = "list_jobs"
    description = "List the resume(s) and job descriptions with their Job # numbers."

    async def run(self, argument: str, ctx: ToolContext) -> ToolOutput:
        jobs = "\n".join(f"- Job #{i}: {d.title}" for i, d in enumerate(_jobs(ctx), 1))
        resumes = ", ".join(d.title for d in _resumes(ctx)) or "none uploaded"
        return ToolOutput(self.name, argument, f"Resume: {resumes}\n{jobs or 'No jobs indexed.'}")


class AnalyzeFit:
    name = "gap_analysis"
    description = (
        "Compare the resume against one job: matched/partial/missing skills and fit score. "
        "Argument: job number like '#2' or a title fragment; 'all' for every job."
    )

    async def run(self, argument: str, ctx: ToolContext) -> ToolOutput:
        resumes = _resumes(ctx)
        if not resumes:
            return ToolOutput(self.name, argument, "No resume has been uploaded.")
        jobs = _pick(_jobs(ctx), argument, "job")
        if not jobs:
            return ToolOutput(self.name, argument, f"No job matches '{argument}'.")
        resume = resumes[-1]
        sections: list[str] = []
        calls: list[LLMCall] = []
        for job in jobs[:4]:
            key = (self.name, resume.owner, resume.source, job.source, ctx.llm.model_for("fast"))
            cached = _CACHE.get(key)
            if cached is None:
                result = await ctx.llm.structured(
                    system=(
                        "You are a technical recruiter. Compare the RESUME to the JOB. List the "
                        "job's key required and preferred skills (max 15) and classify each as "
                        "matched, partial or missing based only on resume evidence."
                    ),
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"RESUME:\n{resume.text[:20000]}\n\nJOB:\n{job.text[:20000]}",
                        )
                    ],
                    schema=GapAnalysis,
                    purpose="tool:gap_analysis",
                )
                cached = _CACHE[key] = result.data
                calls.append(result.call)
            cached = _expect(cached, GapAnalysis)
            rows = "\n".join(f"| {s.skill} | {s.status} | {s.evidence} |" for s in cached.skills)
            sections.append(
                f"### {job.title} — fit {cached.fit_score}/100\n"
                f"| Skill | Status | Evidence |\n|---|---|---|\n{rows}\n\n"
                f"**Experience alignment:** {cached.experience_alignment}\n\n"
                f"**Top gaps:** {'; '.join(cached.top_gaps_to_close)}"
            )
        return ToolOutput(self.name, argument, "\n\n".join(sections), calls)


AGENT_TOOLS: dict[str, list[Tool]] = {
    "docs": [],
    "code": [ListEndpoints(), FindSymbol(), ReadDependencies()],
    "meeting": [ListMeetings(), ExtractActionItems(), ExtractDecisions()],
    "career": [ListJobs(), AnalyzeFit()],
}

# Cheap deterministic tools run on every turn so the model can resolve "Job #2" etc.
ALWAYS_RUN: dict[str, list[str]] = {"meeting": ["list_meetings"], "career": ["list_jobs"]}
