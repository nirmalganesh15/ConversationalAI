"""Deterministic guardrails. Cheap, explainable and unit-tested; the LLM is only
consulted for the ambiguous prompt-injection band (see InjectionVerdict)."""

import re
import time
from collections import defaultdict, deque

from pydantic import BaseModel, Field

from app.core.guardrails.models import GuardrailResult

# --- PII ---------------------------------------------------------------------------
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "mrn": re.compile(r"\bMRN[:#\s]*\d{5,10}\b", re.IGNORECASE),
}


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    for kind, pattern in _PII_PATTERNS.items():
        if pattern.search(text):
            found.append(kind)
            text = pattern.sub(f"[REDACTED_{kind.upper()}]", text)
    return text, found


def check_input_pii(text: str) -> tuple[str, GuardrailResult]:
    redacted, kinds = redact_pii(text)
    if kinds:
        return redacted, GuardrailResult(
            name="pii_input",
            stage="input",
            action="redact",
            detail=f"Redacted {', '.join(kinds)} before sending to the model",
        )
    return text, GuardrailResult(name="pii_input", stage="input", action="pass")


def check_output_pii(text: str) -> tuple[str, GuardrailResult]:
    redacted, kinds = redact_pii(text)
    if kinds:
        return redacted, GuardrailResult(
            name="pii_output",
            stage="output",
            action="redact",
            detail=f"Redacted {', '.join(kinds)} in the answer",
        )
    return text, GuardrailResult(name="pii_output", stage="output", action="pass")


# --- length --------------------------------------------------------------------------
def check_length(text: str, max_chars: int) -> GuardrailResult:
    if not text.strip():
        return GuardrailResult(name="length", stage="input", action="block", detail="Empty query")
    if len(text) > max_chars:
        return GuardrailResult(
            name="length",
            stage="input",
            action="block",
            detail=f"Query is {len(text)} chars; limit is {max_chars}",
        )
    return GuardrailResult(name="length", stage="input", action="pass")


# --- prompt injection -------------------------------------------------------------------
_INJECTION_STRONG = [
    r"ignore (all|any|the)? ?(previous|prior|above) (instructions|prompts?|rules)",
    r"disregard (your|the|all) (instructions|rules|system prompt)",
    r"(reveal|print|show|repeat|leak) (me )?(your|the) (system prompt|instructions|hidden prompt)",
    r"you are now (in )?(dan|developer mode|jailbroken)",
    r"\bjailbreak\b",
    r"act as (an? )?(unfiltered|uncensored)",
]
_INJECTION_WEAK = [
    r"\bsystem prompt\b",
    r"\bpretend (you are|to be)\b",
    r"\boverride\b.*\b(rules|policy|instructions)\b",
    r"\brole ?play\b",
    r"</?(system|assistant|instructions?)>",
]


class InjectionVerdict(BaseModel):
    """Schema for the LLM classifier used only in the ambiguous band."""

    is_injection: bool = Field(description="True if the text tries to subvert instructions")
    reason: str = Field(description="One short sentence")


def injection_heuristic(text: str) -> str:
    """Returns 'strong', 'weak' or 'none'."""
    lowered = text.lower()
    if any(re.search(p, lowered) for p in _INJECTION_STRONG):
        return "strong"
    if any(re.search(p, lowered) for p in _INJECTION_WEAK):
        return "weak"
    return "none"


# --- medical safety ---------------------------------------------------------------------
_PERSONAL_MEDICAL = [
    r"\bshould i (take|stop|start|quit|increase|decrease|double|skip)\b",
    r"\bhow much .{0,40}\bshould i\b",
    r"\b(my|i have|i'm having|i am having) .{0,30}(symptom|pain|dose|dosage|medication|"
    r"prescription|blood pressure|sugar)\b",
    r"\bdo i have\b",
    r"\bam i (sick|diabetic|at risk)\b",
    r"\bdiagnose me\b",
    r"\b(can|could) i (take|mix|combine)\b",
]

MEDICAL_DISCLAIMER = (
    "\n\n> ⚕️ *This information comes from reference documents and is not medical advice. "
    "Please consult a qualified clinician about your own situation.*"
)


def check_medical_safety(text: str) -> GuardrailResult:
    lowered = text.lower()
    if any(re.search(p, lowered) for p in _PERSONAL_MEDICAL):
        return GuardrailResult(
            name="medical_safety",
            stage="input",
            action="flag",
            detail="Personal medical question: answer limited to source content + disclaimer",
        )
    return GuardrailResult(name="medical_safety", stage="input", action="pass")


# --- retrieval confidence ---------------------------------------------------------------
def check_retrieval_confidence(top_score: float | None, threshold: float) -> GuardrailResult:
    if top_score is None:
        return GuardrailResult(
            name="retrieval_confidence",
            stage="retrieval",
            action="block",
            detail="No relevant passages found in the knowledge base",
        )
    if top_score < threshold:
        return GuardrailResult(
            name="retrieval_confidence",
            stage="retrieval",
            action="block",
            detail=f"Best rerank score {top_score:.2f} < threshold {threshold:.2f}",
        )
    return GuardrailResult(
        name="retrieval_confidence",
        stage="retrieval",
        action="pass",
        detail=f"Best rerank score {top_score:.2f}",
    )


# --- citations ------------------------------------------------------------------------
_CITATION = re.compile(r"\[(\d{1,2})\]")
_NOT_FOUND_HINTS = ("couldn't find", "could not find", "not in the provided", "no information")


def citation_report(answer: str, n_sources: int) -> tuple[list[int], list[int], float]:
    """Returns (valid citation ids, invalid ids, fraction of factual sentences cited)."""
    cited = [int(m) for m in _CITATION.findall(answer)]
    valid = sorted({c for c in cited if 1 <= c <= n_sources})
    invalid = sorted({c for c in cited if not 1 <= c <= n_sources})
    sentences = [
        s
        for s in re.split(r"(?<=[.!?])\s+|\n+", answer)
        if len(s.split()) >= 6 and not s.lstrip().startswith(("#", ">"))
    ]
    coverage = sum(1 for s in sentences if _CITATION.search(s)) / len(sentences) if sentences else 0
    return valid, invalid, round(coverage, 2)


def check_citations(answer: str, n_sources: int) -> GuardrailResult:
    valid, invalid, coverage = citation_report(answer, n_sources)
    lowered = answer.lower()
    if any(h in lowered for h in _NOT_FOUND_HINTS):
        return GuardrailResult(
            name="citations", stage="output", action="pass", detail="Model abstained"
        )
    if invalid:
        return GuardrailResult(
            name="citations",
            stage="output",
            action="flag",
            detail=f"Cites non-existent sources {invalid}",
        )
    if not valid:
        return GuardrailResult(
            name="citations", stage="output", action="flag", detail="Answer has no citations"
        )
    return GuardrailResult(
        name="citations",
        stage="output",
        action="pass",
        detail=f"{len(valid)} sources cited, {int(coverage * 100)}% sentence coverage",
    )


# --- rate limiting --------------------------------------------------------------------------
class RateLimiter:
    """In-process sliding window. Production would use Redis or the API gateway."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= self.per_minute:
            return False
        hits.append(now)
        return True
