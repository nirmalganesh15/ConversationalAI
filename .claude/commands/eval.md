---
description: Run the retrieval ablation and compare it with the previous report
argument-hint: [agent]
---

1. Run `cd backend && uv run python -m evals.run retrieval --agent ${1:-all}`.
2. Open the two most recent `backend/evals/reports/retrieval-*.json` files.
3. Produce a short table of passage hit@1 / hit@k / MRR per agent and mode: previous vs new.
4. Flag any metric that dropped by more than 0.05 and point to the likely cause in the diff.
Do not change code in this command.
