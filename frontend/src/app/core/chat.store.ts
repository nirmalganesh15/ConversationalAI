import { Injectable, computed, inject, signal } from '@angular/core';
import { firstValueFrom } from 'rxjs';

import { ApiService } from './api.service';
import { AuthService } from './auth.service';
import { AppConfig, ChatSession, Turn } from './models';

/** Pipeline stages in the order the LangGraph supervisor runs them (for the tracker UI). */
export const PIPELINE = [
  { node: 'input_guard', label: 'Guard' },
  { node: 'plan', label: 'Plan' },
  { node: 'tools', label: 'Tools' },
  { node: 'retrieve', label: 'Retrieve' },
  { node: 'generate', label: 'Generate' },
  { node: 'output_guard', label: 'Check' },
  { node: 'evaluate', label: 'Evaluate' },
] as const;

interface SseEvent {
  event: string;
  data: any; // eslint-disable-line @typescript-eslint/no-explicit-any
}

/**
 * Signal-based store for the workspace: config, sessions, the active conversation and the
 * live stream. Streaming uses fetch + a manual SSE parser because EventSource cannot POST.
 */
@Injectable({ providedIn: 'root' })
export class ChatStore {
  private readonly api = inject(ApiService);
  private readonly auth = inject(AuthService);

  readonly config = signal<AppConfig | null>(null);
  readonly sessions = signal<ChatSession[]>([]);
  readonly sessionId = signal<string | null>(null);
  readonly turns = signal<Turn[]>([]);
  readonly agent = signal<string>('docs');
  readonly provider = signal<string>('claude');
  readonly model = signal<string>('');
  readonly selectedTurn = signal<number>(-1);
  readonly highlightedSource = signal<number | null>(null);
  readonly busy = signal(false);
  private abort: AbortController | null = null;

  readonly agentInfo = computed(() => this.config()?.agents.find((a) => a.name === this.agent()));
  readonly providerInfo = computed(() =>
    this.config()?.providers.find((p) => p.key === this.provider()),
  );
  readonly inspected = computed<Turn | null>(() => {
    const turns = this.turns();
    const index = this.selectedTurn();
    return turns[index >= 0 ? index : turns.length - 1] ?? null;
  });

  async init(): Promise<void> {
    const config = await firstValueFrom(this.api.config());
    this.config.set(config);
    if (!config.agents.some((a) => a.name === this.agent())) {
      this.agent.set(config.agents[0]?.name ?? 'docs');
    }
    const provider =
      config.providers.find((p) => p.available && p.key === 'claude') ??
      config.providers.find((p) => p.available);
    if (provider) this.selectProvider(provider.key);
    await this.refreshSessions();
  }

  selectProvider(key: string): void {
    const info = this.config()?.providers.find((p) => p.key === key);
    if (!info) return;
    this.provider.set(key);
    this.model.set(info.default_model);
  }

  async refreshSessions(): Promise<void> {
    this.sessions.set(await firstValueFrom(this.api.sessions()));
  }

  newChat(agent?: string): void {
    this.stop();
    if (agent) this.agent.set(agent);
    this.sessionId.set(null);
    this.turns.set([]);
    this.selectedTurn.set(-1);
  }

  async openSession(session: ChatSession): Promise<void> {
    this.stop();
    const detail = await firstValueFrom(this.api.session(session.id));
    this.agent.set(session.agent);
    this.sessionId.set(session.id);
    this.turns.set(detail.turns);
    this.selectedTurn.set(-1);
  }

  async deleteSession(id: string): Promise<void> {
    await firstValueFrom(this.api.deleteSession(id));
    if (this.sessionId() === id) this.newChat();
    await this.refreshSessions();
  }

  stop(): void {
    this.abort?.abort();
    this.abort = null;
    this.busy.set(false);
  }

  async ask(question: string): Promise<void> {
    if (!question.trim() || this.busy()) return;
    const turn: Turn = {
      question,
      answer: '',
      agent: this.agent(),
      provider: this.provider(),
      model: this.model(),
      sources: [],
      tools: [],
      guardrails: [],
      evals: null,
      usage: null,
      search_query: '',
      streaming: true,
      completedNodes: [],
    };
    this.turns.update((t) => [...t, turn]);
    this.selectedTurn.set(-1);
    this.busy.set(true);
    this.abort = new AbortController();

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.auth.token()}`,
        },
        body: JSON.stringify({
          question,
          agent: this.agent(),
          provider: this.provider(),
          model: this.model(),
          session_id: this.sessionId(),
        }),
        signal: this.abort.signal,
      });
      if (response.status === 401) {
        this.auth.logout();
        return;
      }
      if (!response.ok || !response.body) {
        const detail = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(detail.detail ?? `HTTP ${response.status}`);
      }
      for await (const event of readSse(response.body)) {
        this.apply(event);
      }
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        this.patchLast({ error: (error as Error).message });
      }
    } finally {
      this.patchLast({ streaming: false, activeNode: undefined });
      this.busy.set(false);
      this.abort = null;
      void this.refreshSessions();
    }
  }

  private apply({ event, data }: SseEvent): void {
    switch (event) {
      case 'session':
        this.sessionId.set(data.id);
        break;
      case 'status':
        this.patchLast((t) => ({
          activeNode: data.node,
          completedNodes: t.activeNode
            ? [...(t.completedNodes ?? []), t.activeNode]
            : t.completedNodes,
        }));
        break;
      case 'token':
        this.patchLast((t) => ({ answer: t.answer + data.text }));
        break;
      case 'plan':
        this.patchLast({ search_query: data.search_query });
        break;
      case 'tool':
        this.patchLast((t) => ({ tools: [...t.tools, data] }));
        break;
      case 'sources':
        this.patchLast({ sources: data.items });
        break;
      case 'guardrail':
        this.patchLast((t) => ({ guardrails: [...t.guardrails, data.item] }));
        break;
      case 'eval':
        this.patchLast({ evals: data.metrics });
        break;
      case 'usage':
        this.patchLast({
          usage: { calls: data.calls, totals: data.totals, timings: data.timings },
        });
        break;
      case 'done':
        // The final answer is authoritative (output guardrails may have redacted it).
        this.patchLast((t) => ({
          answer: data.answer,
          completedNodes: [...(t.completedNodes ?? []), ...(t.activeNode ? [t.activeNode] : [])],
        }));
        break;
      case 'error':
        this.patchLast({ error: data.message });
        break;
    }
  }

  private patchLast(patch: Partial<Turn> | ((turn: Turn) => Partial<Turn>)): void {
    this.turns.update((turns) => {
      if (!turns.length) return turns;
      const last = turns[turns.length - 1];
      const update = typeof patch === 'function' ? patch(last) : patch;
      return [...turns.slice(0, -1), { ...last, ...update }];
    });
  }
}

/** Minimal Server-Sent Events parser over a fetch body stream. */
export async function* readSse(body: ReadableStream<Uint8Array>): AsyncGenerator<SseEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
    let boundary: number;
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = 'message';
      const data: string[] = [];
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
      }
      if (data.length) {
        try {
          yield { event, data: JSON.parse(data.join('\n')) };
        } catch {
          /* keep-alive comments / non-JSON pings */
        }
      }
    }
  }
}
