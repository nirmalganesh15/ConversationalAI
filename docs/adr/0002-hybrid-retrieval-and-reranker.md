# ADR-0002 — Hybrid retrieval with stemming, RRF and a CPU reranker

**Status:** accepted · **Context:** clinical text mixes paraphrasable prose ("high blood pressure"
vs "hypertension") with exact tokens that embeddings blur (doses, `eGFR`, NCT ids). The app runs
on a CPU-only Space.

## Decision
- Lexical BM25 with Snowball stemming, stopwords and an abbreviation passthrough; identifier
  analyzer for code. Dense bge-small. Reciprocal Rank Fusion (k=60). Cross-encoder rerank of the
  top-30, keep 6 within a token budget.
- Reranker default: `cross-encoder/ms-marco-MiniLM-L-6-v2`; `BAAI/bge-reranker-base` is a config
  switch.
- Rerank scores pass through one explicit sigmoid (`activation_fn`), because models differ in
  their default activation — without it, bge scores were double-sigmoided into 0.5–0.73 and the
  confidence gate was meaningless.
- The abstain threshold is **per agent** (docs 0.05, meeting/career 0.005, code 0.0005): ms-marco
  rates relevant code ~0.002 while off-topic questions score ~0.0001.

## Evidence (`make eval-retrieval`, passage hit@1, top-6)
| | Dense | BM25 | Hybrid | +MiniLM | +bge-base |
|---|---|---|---|---|---|
| Docs (23) | 0.96 | 0.78 | 0.83 | 0.78 | 0.87 |
| Code (6) | 0.67 | 0.33 | 0.67 | 0.83 | 0.67 |
| Meeting (7) | 1.00 | 0.86 | 0.86 | 1.00 | – |
| Career (6) | 0.50 | 0.17 | 0.17 | 0.33 | 0.50 |
| p50 latency | 14 ms | 1 ms | 14 ms | ~650 ms | ~4.6 s |

## Consequences
+ Robust to both paraphrase and exact-term queries; per-stage scores are explainable in the UI.
+ The confidence gate avoids an LLM call (cost and hallucination risk) when nothing relevant exists.
− On this small golden set dense-only is competitive for docs; the hybrid is kept for larger
  corpora and identifier-heavy queries, and the decision should be re-checked on real traffic.
− ~0.6 s CPU rerank per query; on GPU or with a served reranker this drops to tens of ms.
