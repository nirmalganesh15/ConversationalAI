"""Per-answer quality metrics shown live in the Inspector.

One judge call (fast tier) scores three RAG-triad metrics together - cheaper and faster
than three calls, at the cost of some cross-metric correlation (documented trade-off).
"""

from pydantic import BaseModel, Field

from app.core.guardrails.checks import citation_report
from app.core.llm.base import ChatMessage, LLMCall, LLMProvider
from app.core.retrieval.models import RetrievedChunk

_JUDGE_SYSTEM = """You are a strict evaluator of retrieval-augmented answers.
Score each metric from 0.0 to 1.0:
- faithfulness: fraction of the answer's claims that are directly supported by the CONTEXT.
  Claims not found in the context lower the score even if they are true in general.
- answer_relevance: how directly and completely the answer addresses the QUESTION.
- context_precision: fraction of the CONTEXT passages that are relevant to the QUESTION.
If the answer correctly says the information is not in the context, faithfulness is 1.0.
Be terse in the rationale (max 25 words)."""


class JudgeScores(BaseModel):
    faithfulness: float = Field(ge=0, le=1)
    answer_relevance: float = Field(ge=0, le=1)
    context_precision: float = Field(ge=0, le=1)
    rationale: str


class EvalMetrics(BaseModel):
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    citation_coverage: float = 0.0
    citations_valid: int = 0
    top_retrieval_score: float | None = None
    rationale: str = ""
    judge_model: str | None = None


def deterministic_metrics(answer: str, contexts: list[RetrievedChunk]) -> EvalMetrics:
    valid, _invalid, coverage = citation_report(answer, len(contexts))
    return EvalMetrics(
        citation_coverage=coverage,
        citations_valid=len(valid),
        top_retrieval_score=contexts[0].confidence if contexts else None,
    )


def format_contexts(contexts: list[RetrievedChunk], max_chars: int = 1200) -> str:
    return "\n\n".join(f"[{i}] {c.chunk.text[:max_chars]}" for i, c in enumerate(contexts, 1))


async def judge_answer(
    judge: LLMProvider, question: str, answer: str, contexts: list[RetrievedChunk]
) -> tuple[EvalMetrics, LLMCall]:
    metrics = deterministic_metrics(answer, contexts)
    prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{format_contexts(contexts)}\n\nANSWER:\n{answer}"
    result = await judge.structured(
        system=_JUDGE_SYSTEM,
        messages=[ChatMessage(role="user", content=prompt)],
        schema=JudgeScores,
        purpose="eval_judge",
        tier="fast",
    )
    scores = result.data
    metrics.faithfulness = round(scores.faithfulness, 2)
    metrics.answer_relevance = round(scores.answer_relevance, 2)
    metrics.context_precision = round(scores.context_precision, 2)
    metrics.rationale = scores.rationale
    metrics.judge_model = result.call.model
    return metrics, result.call
