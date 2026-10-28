"""Agent specs - the 'domain pack' that turns the shared RAG core into a specific agent.

Adding a fifth agent = add an AgentName, a spec here, a prompt file, and (optionally)
tools in its module. Nothing in the core changes.
"""

from dataclasses import dataclass
from pathlib import Path

from app.core.ingestion.chunkers import (
    Chunker,
    CodeChunker,
    RecursiveChunker,
    SectionChunker,
    TranscriptChunker,
)
from app.core.retrieval.analyzer import Analyzer, CodeAnalyzer, StemmingAnalyzer
from app.core.security.policy import AgentName

_PROMPTS = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class AgentSpec:
    name: AgentName
    title: str
    description: str
    icon: str
    accepts: tuple[str, ...]
    analyzer: Analyzer
    chunker: Chunker
    example_questions: tuple[str, ...]
    # Rerank score below which the agent abstains. Calibrated per domain: the ms-marco
    # reranker scores relevant code far lower than relevant prose (see docs/adr/0002).
    min_confidence: float = 0.05

    @property
    def system_prompt(self) -> str:
        shared = (_PROMPTS / "_shared.md").read_text(encoding="utf-8")
        specific = (_PROMPTS / f"{self.name.value}.md").read_text(encoding="utf-8")
        return f"{specific}\n\n{shared}"


SPECS: dict[AgentName, AgentSpec] = {
    AgentName.DOCS: AgentSpec(
        name=AgentName.DOCS,
        title="Clinical Docs",
        description="Ask questions across healthcare documents - drug labels, guidelines, "
        "fact sheets, trial records.",
        icon="description",
        accepts=(".pdf", ".md", ".txt", ".docx"),
        analyzer=StemmingAnalyzer(),
        chunker=RecursiveChunker(target_tokens=400, overlap_tokens=60),
        example_questions=(
            "What is the maximum daily dose of metformin?",
            "Which patients should not take lisinopril?",
            "How is high blood pressure diagnosed?",
        ),
    ),
    AgentName.CODE: AgentSpec(
        name=AgentName.CODE,
        title="Code Docs",
        description="Understand a codebase: architecture, where things live, endpoints, "
        "dependencies.",
        icon="code",
        accepts=(".zip", "github-url"),
        analyzer=CodeAnalyzer(),
        chunker=CodeChunker(max_lines=80),
        min_confidence=0.0005,
        example_questions=(
            "How does the hybrid retriever combine BM25 and vector search?",
            "Which API endpoints exist and what roles can call them?",
            "Where is role-based access control enforced?",
        ),
    ),
    AgentName.MEETING: AgentSpec(
        name=AgentName.MEETING,
        title="Meeting Intelligence",
        description="Decisions, action items and who-said-what across meeting transcripts.",
        icon="groups",
        accepts=(".txt", ".vtt", ".md"),
        analyzer=StemmingAnalyzer(),
        chunker=TranscriptChunker(target_tokens=350, overlap_turns=1),
        min_confidence=0.005,
        example_questions=(
            "What decisions were made in the trial steering committee?",
            "List all action items with owners and due dates.",
            "What concerns did the safety officer raise?",
        ),
    ),
    AgentName.CAREER: AgentSpec(
        name=AgentName.CAREER,
        title="Career Intelligence",
        description="Compare a resume against job descriptions: fit, gaps, interview prep.",
        icon="work",
        accepts=(".pdf", ".md", ".txt", ".docx"),
        analyzer=StemmingAnalyzer(),
        chunker=SectionChunker(max_tokens=450),
        min_confidence=0.005,
        example_questions=(
            "What skills am I missing for the Forward Deployed Engineer role?",
            "How does my experience align with Job #2?",
            "Prepare me for an interview for the best-fit role.",
        ),
    ),
}
