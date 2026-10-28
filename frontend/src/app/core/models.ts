/** Types mirroring the FastAPI contracts (backend/app/api/*). */

export interface User {
  username: string;
  role: 'admin' | 'staff';
  allowed_agents: string[];
  is_admin: boolean;
}

export interface AgentInfo {
  name: string;
  title: string;
  description: string;
  icon: string;
  accepts: string[];
  examples: string[];
}

export interface ProviderInfo {
  key: string;
  label: string;
  models: string[];
  default_model: string;
  available: boolean;
}

export interface AppConfig {
  agents: AgentInfo[];
  providers: ProviderInfo[];
  retrieval: { embedding_model: string; reranker_model: string | null; top_k: number };
  status: { seed: string };
}

export interface ChatSession {
  id: string;
  agent: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChunkMeta {
  source: string;
  title?: string;
  page?: number;
  section?: string;
  path?: string;
  symbol?: string;
  line_start?: number;
  line_end?: number;
  ts_start?: string;
  ts_end?: string;
  speakers?: string;
  doc_role?: string;
  [key: string]: unknown;
}

export interface SourceItem {
  chunk: { id: string; text: string; metadata: ChunkMeta };
  bm25_rank: number | null;
  dense_rank: number | null;
  rrf_score: number;
  rerank_score: number | null;
  confidence: number;
}

export type GuardAction = 'pass' | 'flag' | 'redact' | 'block';

export interface Guardrail {
  name: string;
  stage: 'input' | 'retrieval' | 'output';
  action: GuardAction;
  detail: string;
}

export interface EvalMetrics {
  faithfulness: number | null;
  answer_relevance: number | null;
  context_precision: number | null;
  citation_coverage: number;
  citations_valid: number;
  top_retrieval_score: number | null;
  rationale: string;
  judge_model: string | null;
}

export interface LlmCall {
  purpose: string;
  provider: string;
  model: string;
  usage: {
    input_tokens: number;
    output_tokens: number;
    cache_read_tokens: number;
    cache_write_tokens: number;
  };
  cost_usd: number;
  latency_ms: number;
  stop_reason: string | null;
}

export interface UsageInfo {
  calls: LlmCall[];
  totals: {
    input_tokens: number;
    output_tokens: number;
    cache_read_tokens: number;
    cost_usd: number;
  };
  timings: Record<string, number>;
}

export interface ToolRun {
  name: string;
  argument: string;
  text?: string;
  output?: string;
}

/** One question/answer exchange with everything the Inspector shows about it. */
export interface Turn {
  question: string;
  answer: string;
  agent: string;
  provider: string;
  model: string;
  sources: SourceItem[];
  tools: ToolRun[];
  guardrails: Guardrail[];
  evals: EvalMetrics | null;
  usage: UsageInfo | null;
  search_query: string;
  streaming?: boolean;
  activeNode?: string;
  completedNodes?: string[];
  error?: string;
}

export interface SourceRow {
  source: string;
  title: string;
  doc_type: string;
  chunks: number;
  shared: boolean;
  deletable: boolean;
}

export interface IngestReport {
  agent: string;
  documents: number;
  chunks: number;
  sources: string[];
}
