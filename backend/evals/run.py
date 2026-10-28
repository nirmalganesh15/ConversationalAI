"""Offline evaluation harness.

  python -m evals.run retrieval [--agent docs|code|meeting|career|all]
      Retrieval ablation - no LLM needed. hit@1 / hit@k / MRR / latency for
      dense vs bm25 vs hybrid vs hybrid_rerank, through the production retriever.

  python -m evals.run e2e --agent docs --provider claude|open [--model M] [--limit N]
      Full graph per case: correctness, abstention, guardrails, judge metrics, latency, cost.
      Exits non-zero when quality gates fail, so CI can block a regression.

Reports land in evals/reports/*.json (rendered by the admin Evals dashboard) and *.md.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from langgraph.checkpoint.memory import InMemorySaver

from app.container import Container, build_container
from app.core.config import get_settings
from app.core.ingestion.pipeline import SYSTEM_OWNER
from app.graph.state import GraphDeps, new_turn
from app.graph.supervisor import build_graph
from app.seed import seed

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
REPORTS = ROOT / "reports"
MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]
AGENTS = ["docs", "code", "meeting", "career"]

# Quality gates for `e2e` (CI fails below these).
GATES = {
    "retrieval_hit_rate": 0.8,
    "answer_correct_rate": 0.7,
    "faithfulness_mean": 0.75,
    "safety_pass_rate": 1.0,
}


def load_cases(agent: str) -> list[dict[str, Any]]:
    data = yaml.safe_load((DATASETS / f"{agent}.yaml").read_text(encoding="utf-8"))
    return list(data["cases"])


def source_of(meta: dict[str, Any], agent: str) -> str:
    # Code chunks are identified by repo-relative path; everything else by file name.
    return str(meta.get("path" if agent == "code" else "source", ""))


def first_hit_rank(sources: list[str], expected: list[str]) -> int | None:
    for rank, source in enumerate(sources, start=1):
        if any(source.endswith(e) or e.endswith(source) for e in expected):
            return rank
    return None


def first_passage_rank(
    found: list[Any], agent: str, expected: list[str], must_include: list[str]
) -> int | None:
    """Stricter than a document hit: the chunk itself must contain the answer text."""
    for rank, result in enumerate(found, start=1):
        text = result.chunk.text.lower()
        right_doc = first_hit_rank([source_of(result.chunk.metadata, agent)], expected)
        if right_doc and all(m.lower() in text for m in must_include):
            return rank
    return None


def _rate(ranks: list[int | None], k: int) -> float:
    return round(sum(1 for r in ranks if r is not None and r <= k) / len(ranks), 3)


# --- retrieval ablation ------------------------------------------------------------------
def run_retrieval(container: Container, agents: list[str]) -> dict[str, Any]:
    top_k = container.settings.context_top_k
    results: dict[str, Any] = {}
    for agent in agents:
        cases = [c for c in load_cases(agent) if c.get("expected_sources")]
        retriever = container.kb.retriever(agent)
        per_mode: dict[str, Any] = {}
        for mode in MODES:
            doc_ranks: list[int | None] = []
            passage_ranks: list[int | None] = []
            latencies: list[float] = []
            rows: list[dict[str, Any]] = []
            for case in cases:
                start = time.perf_counter()
                found = retriever.search(case["question"], {SYSTEM_OWNER}, mode=mode, top_k=top_k)  # type: ignore[arg-type]
                latencies.append((time.perf_counter() - start) * 1000)
                sources = [source_of(r.chunk.metadata, agent) for r in found]
                doc_ranks.append(first_hit_rank(sources, case["expected_sources"]))
                passage_ranks.append(
                    first_passage_rank(
                        found, agent, case["expected_sources"], case.get("must_include") or []
                    )
                )
                rows.append(
                    {
                        "id": case["id"],
                        "category": case["category"],
                        "doc_rank": doc_ranks[-1],
                        "passage_rank": passage_ranks[-1],
                    }
                )
            per_mode[mode] = {
                "hit@1": _rate(doc_ranks, 1),
                f"hit@{top_k}": _rate(doc_ranks, top_k),
                "passage_hit@1": _rate(passage_ranks, 1),
                f"passage_hit@{top_k}": _rate(passage_ranks, top_k),
                "passage_mrr": round(statistics.mean(1 / r if r else 0 for r in passage_ranks), 3),
                "latency_ms_p50": round(statistics.median(latencies), 1),
                "cases": rows,
            }
        results[agent] = {"n": len(cases), "modes": per_mode}
    return {
        "kind": "retrieval",
        "top_k": top_k,
        "embedding_model": container.settings.embedding_model,
        "reranker_model": container.settings.reranker_model,
        "agents": results,
    }


# --- end-to-end -----------------------------------------------------------------------------
async def run_e2e(
    container: Container, agent: str, provider: str, model: str | None, limit: int | None
) -> dict[str, Any]:
    graph = build_graph(
        GraphDeps(kb=container.kb, providers=container.providers, settings=container.settings),
        InMemorySaver(),
    )
    llm = container.providers.get(provider, model)
    cases = load_cases(agent)[:limit] if limit else load_cases(agent)
    rows: list[dict[str, Any]] = []
    for case in cases:
        start = time.perf_counter()
        state = await graph.ainvoke(
            new_turn(
                user_id="eval",
                role="admin",
                agent=agent,
                provider=provider,
                model=model or llm.model_for("main"),
                question=case["question"],
            ),
            {"configurable": {"thread_id": f"eval:{uuid.uuid4().hex}"}},
        )
        latency = (time.perf_counter() - start) * 1000
        answer: str = state["answer"]
        lowered = answer.lower()
        sources = [source_of(c["chunk"]["metadata"], agent) for c in state["contexts"]]
        tripped = {g["name"] for g in state["guardrails"] if g["action"] != "pass"}
        evals = state.get("evals") or {}
        row: dict[str, Any] = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": answer[:600],
            "latency_ms": round(latency),
            "cost_usd": round(sum(c["cost_usd"] for c in state["llm_calls"]), 5),
            "tokens": sum(
                c["usage"]["input_tokens"] + c["usage"]["output_tokens"] for c in state["llm_calls"]
            ),
            "guardrails_tripped": sorted(tripped),
            "faithfulness": evals.get("faithfulness"),
            "answer_relevance": evals.get("answer_relevance"),
            "context_precision": evals.get("context_precision"),
        }
        if case.get("expected_sources"):
            row["retrieval_hit"] = first_hit_rank(sources, case["expected_sources"]) is not None
            must = case.get("must_include") or []
            row["answer_correct"] = all(m.lower() in lowered for m in must)
        if case.get("expect_abstain"):
            row["safety_pass"] = "couldn't find" in lowered or "retrieval_confidence" in tripped
        if case.get("expect_guardrail"):
            row["safety_pass"] = case["expect_guardrail"] in tripped
        if case.get("expect_blocked"):
            row["safety_pass"] = bool(state["blocked"]) and not state["contexts"]
        rows.append(row)
        print(
            f"  {case['id']:12s} {row['latency_ms']:6d}ms  "
            f"{'ok' if row.get('answer_correct', row.get('safety_pass')) else 'FAIL'}"
        )

    def rate(key: str) -> float | None:
        values = [r[key] for r in rows if key in r]
        return round(sum(values) / len(values), 3) if values else None

    faith = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    relevance = [r["answer_relevance"] for r in rows if r["answer_relevance"] is not None]
    summary = {
        "n": len(rows),
        "retrieval_hit_rate": rate("retrieval_hit"),
        "answer_correct_rate": rate("answer_correct"),
        "safety_pass_rate": rate("safety_pass"),
        "faithfulness_mean": round(statistics.mean(faith), 3) if faith else None,
        "answer_relevance_mean": round(statistics.mean(relevance), 3) if relevance else None,
        "latency_ms_p50": round(statistics.median(r["latency_ms"] for r in rows)),
        "latency_ms_p95": round(sorted(r["latency_ms"] for r in rows)[int(len(rows) * 0.95) - 1]),
        "cost_usd_total": round(sum(r["cost_usd"] for r in rows), 4),
    }
    gates = {k: (summary.get(k) is None or summary[k] >= v) for k, v in GATES.items()}
    return {
        "kind": "e2e",
        "agent": agent,
        "provider": provider,
        "model": model or llm.model_for("main"),
        "summary": summary,
        "gates": gates,
        "passed": all(gates.values()),
        "cases": rows,
    }


# --- reporting --------------------------------------------------------------------------------
def write_report(report: dict[str, Any]) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = {"id": f"{report['kind']}-{stamp}", "created_at": stamp, **report}
    path = REPORTS / f"{report['id']}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    path.with_suffix(".md").write_text(to_markdown(report), encoding="utf-8")
    return path


def to_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Eval report `{report['id']}`", ""]
    if report["kind"] == "retrieval":
        k = report["top_k"]
        lines += [
            f"Embedding `{report['embedding_model']}`, reranker `{report['reranker_model']}`, "
            f"top-k {k}.",
            "",
        ]
        for agent, data in report["agents"].items():
            lines += [
                f"## {agent} ({data['n']} questions)",
                "",
                f"| Mode | doc hit@1 | doc hit@{k} | passage hit@1 | passage hit@{k} "
                "| passage MRR | p50 latency |",
                "|---|---|---|---|---|---|---|",
            ]
            for mode, m in data["modes"].items():
                lines.append(
                    f"| {mode} | {m['hit@1']} | {m[f'hit@{k}']} | {m['passage_hit@1']} "
                    f"| {m[f'passage_hit@{k}']} | {m['passage_mrr']} | {m['latency_ms_p50']} ms |"
                )
            lines.append("")
    else:
        lines += [
            f"Agent `{report['agent']}`, model `{report['model']}` - "
            f"**{'PASSED' if report['passed'] else 'FAILED'}**",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        lines += [f"| {k} | {v} |" for k, v in report["summary"].items()]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("kind", choices=["retrieval", "e2e"])
    parser.add_argument("--agent", default="all", choices=[*AGENTS, "all"])
    parser.add_argument("--provider", default="claude")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    container = build_container(get_settings())
    seed(container, Path(__file__).resolve().parents[2])
    agents = AGENTS if args.agent == "all" else [args.agent]

    if args.kind == "retrieval":
        report = run_retrieval(container, agents)
        path = write_report(report)
        print(to_markdown({"id": path.stem, **report}))
        return 0

    exit_code = 0
    for agent in agents:
        print(f"[{agent}] running {args.provider}")
        report = asyncio.run(run_e2e(container, agent, args.provider, args.model, args.limit))
        path = write_report(report)
        print(to_markdown({"id": path.stem, **report}))
        exit_code |= 0 if report["passed"] else 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
