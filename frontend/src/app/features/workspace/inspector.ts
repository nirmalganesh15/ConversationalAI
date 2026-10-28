import { DecimalPipe } from '@angular/common';
import { Component, ElementRef, computed, effect, inject, model, signal } from '@angular/core';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MarkdownComponent } from 'ngx-markdown';

import { ChatStore } from '../../core/chat.store';
import { ChunkMeta, Guardrail } from '../../core/models';
import { Gauge } from './gauge';

export type InspectorTab = 'sources' | 'guardrails' | 'evals' | 'usage' | 'trace';

const NODE_LABELS: Record<string, string> = {
  input_guard: 'Input guardrails',
  plan: 'Planner (query rewrite + tools)',
  tools: 'Tool execution',
  retrieve: 'Hybrid retrieval + rerank',
  generate: 'Answer generation',
  output_guard: 'Output guardrails',
  evaluate: 'LLM-as-judge',
  summarize: 'Memory summarisation',
};

/** Right-hand panel: everything about one answer - the "glass box" view. */
@Component({
  selector: 'app-inspector',
  imports: [Gauge, DecimalPipe, MatTooltipModule, MarkdownComponent],
  templateUrl: './inspector.html',
  styleUrl: './inspector.scss',
})
export class Inspector {
  protected readonly store = inject(ChatStore);
  private readonly host = inject(ElementRef<HTMLElement>);
  readonly tab = model<InspectorTab>('sources');
  protected readonly expanded = signal<Set<number>>(new Set());

  protected readonly turn = this.store.inspected;
  protected readonly tabs: { id: InspectorTab; icon: string; label: string }[] = [
    { id: 'sources', icon: 'library_books', label: 'Sources' },
    { id: 'guardrails', icon: 'shield', label: 'Guardrails' },
    { id: 'evals', icon: 'verified', label: 'Evals' },
    { id: 'usage', icon: 'token', label: 'Usage' },
    { id: 'trace', icon: 'timeline', label: 'Trace' },
  ];

  protected readonly guardGroups = computed(() => {
    const items = this.turn()?.guardrails ?? [];
    return (['input', 'retrieval', 'output'] as const)
      .map((stage) => ({ stage, items: items.filter((g) => g.stage === stage) }))
      .filter((g) => g.items.length);
  });

  protected readonly tripCount = computed(
    () => (this.turn()?.guardrails ?? []).filter((g) => g.action !== 'pass').length,
  );

  protected readonly waterfall = computed(() => {
    const timings = this.turn()?.usage?.timings ?? {};
    const entries = Object.entries(timings);
    const total = entries.reduce((sum, [, ms]) => sum + ms, 0) || 1;
    let offset = 0;
    return entries.map(([node, ms]) => {
      const row = {
        node,
        label: NODE_LABELS[node] ?? node,
        ms,
        left: (offset / total) * 100,
        width: Math.max((ms / total) * 100, 0.8),
      };
      offset += ms;
      return row;
    });
  });

  protected readonly totalMs = computed(() =>
    this.waterfall().reduce((sum, row) => sum + row.ms, 0),
  );

  constructor() {
    // A clicked citation switches to Sources and scrolls the passage into view.
    effect(() => {
      const index = this.store.highlightedSource();
      if (index === null) return;
      this.tab.set('sources');
      this.expanded.update((s) => new Set(s).add(index - 1));
      queueMicrotask(() =>
        this.host.nativeElement
          .querySelector(`#src-${index}`)
          ?.scrollIntoView({ behavior: 'smooth', block: 'center' }),
      );
    });
  }

  protected label(meta: ChunkMeta): string {
    if (meta.line_start)
      return `${meta.path} :: ${meta.symbol} (L${meta.line_start}–${meta.line_end})`;
    let label = String(meta.title ?? meta.source);
    if (meta.page) label += `, p.${meta.page}`;
    if (meta.section) label += ` — ${meta.section}`;
    if (meta.ts_start) label += ` [${meta.ts_start}–${meta.ts_end}]`;
    return label;
  }

  protected toggle(index: number): void {
    this.expanded.update((s) => {
      const next = new Set(s);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }

  protected icon(g: Guardrail): string {
    return { pass: 'check_circle', flag: 'flag', redact: 'visibility_off', block: 'block' }[
      g.action
    ];
  }

  protected pretty(name: string): string {
    return name.replace(/_/g, ' ');
  }
}
