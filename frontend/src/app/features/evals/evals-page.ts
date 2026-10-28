import { DecimalPipe, KeyValuePipe } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { firstValueFrom } from 'rxjs';

import { ApiService, EvalReport } from '../../core/api.service';

const MODE_LABELS: Record<string, string | undefined> = {
  dense: 'Dense only',
  bm25: 'BM25 (stemmed)',
  hybrid: 'Hybrid (RRF)',
  hybrid_rerank: 'Hybrid + rerank',
};

/** Admin-only view of offline eval runs produced by `python -m evals.run`. */
@Component({
  selector: 'app-evals-page',
  imports: [RouterLink, DecimalPipe, KeyValuePipe],
  templateUrl: './evals-page.html',
  styleUrl: './evals-page.scss',
})
export class EvalsPage {
  private readonly api = inject(ApiService);
  protected readonly reports = signal<EvalReport[]>([]);
  protected readonly selectedId = signal<string | null>(null);
  protected readonly loading = signal(true);
  protected readonly modeLabels = MODE_LABELS;

  protected readonly selected = computed<EvalReport | null>(
    () => this.reports().find((r) => r['id'] === this.selectedId()) ?? this.reports()[0] ?? null,
  );

  constructor() {
    firstValueFrom(this.api.evalReports())
      .then((reports) => this.reports.set(reports))
      .finally(() => this.loading.set(false));
  }

  protected modes(agentData: EvalReport): { mode: string; m: EvalReport }[] {
    return Object.entries(agentData['modes']).map(([mode, m]) => ({ mode, m: m as EvalReport }));
  }

  protected best(agentData: EvalReport, key: string): number {
    return Math.max(...this.modes(agentData).map(({ m }) => m[key] as number));
  }

  protected pct(value: number | null | undefined): string {
    return value === null || value === undefined ? '–' : `${Math.round(value * 100)}%`;
  }

  protected summary(
    report: EvalReport,
  ): { key: string; label: string; display: string; failed: boolean }[] {
    return Object.entries(report['summary'] as Record<string, number | null>).map(
      ([key, value]) => ({
        key,
        label: key.replace(/_/g, ' '),
        display:
          key.endsWith('_rate') || key.endsWith('_mean') ? this.pct(value) : String(value ?? '–'),
        failed: report['gates']?.[key] === false,
      }),
    );
  }
}
