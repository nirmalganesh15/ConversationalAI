"""Context engineering: what goes into the generation prompt, in what order, under a budget.

Order is chosen for prompt caching and attention:
  system (frozen per agent, cached) -> recent turns -> [summary, tool results,
  numbered passages, question] in the final user message.
"""

from typing import Any

from app.core.ingestion.chunkers import approx_tokens
from app.core.llm.base import ChatMessage


def source_label(meta: dict[str, Any]) -> str:
    label = str(meta.get("title") or meta.get("source", "source"))
    if meta.get("page"):
        label += f", p.{meta['page']}"
    if meta.get("section"):
        label += f" — {meta['section']}"
    if meta.get("line_start"):
        lines = f"L{meta['line_start']}-{meta.get('line_end')}"
        label = f"{meta.get('path', label)} :: {meta.get('symbol')} ({lines})"
    if meta.get("ts_start"):
        label += f" [{meta['ts_start']}-{meta.get('ts_end')}]"
    return label


def fit_contexts(contexts: list[dict[str, Any]], budget_tokens: int) -> list[dict[str, Any]]:
    """Keep the best-ranked passages that fit the budget (never fewer than one)."""
    kept: list[dict[str, Any]] = []
    used = 0
    for ctx in contexts:
        cost = approx_tokens(ctx["chunk"]["text"])
        if kept and used + cost > budget_tokens:
            break
        kept.append(ctx)
        used += cost
    return kept


def build_messages(
    *,
    question: str,
    history: list[dict[str, str]],
    summary: str,
    contexts: list[dict[str, Any]],
    tool_outputs: list[dict[str, str]],
    history_turns: int,
) -> list[ChatMessage]:
    recent = history[-history_turns * 2 :] if history_turns else []
    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in recent]  # type: ignore[arg-type]

    parts: list[str] = []
    if summary:
        parts.append(f"CONVERSATION SUMMARY (older turns):\n{summary}")
    if tool_outputs:
        rendered = "\n\n".join(
            f'<tool name="{t["name"]}" argument="{t["argument"]}">\n{t["text"]}\n</tool>'
            for t in tool_outputs
        )
        parts.append(f"TOOL RESULTS:\n{rendered}")
    if contexts:
        passages = "\n\n".join(
            f"[{i}] ({source_label(c['chunk']['metadata'])})\n{c['chunk']['text']}"
            for i, c in enumerate(contexts, 1)
        )
        parts.append(f"CONTEXT:\n{passages}")
    else:
        parts.append("CONTEXT:\n(no passages retrieved)")
    parts.append(f"QUESTION:\n{question}")
    messages.append(ChatMessage(role="user", content="\n\n".join(parts)))
    return messages
