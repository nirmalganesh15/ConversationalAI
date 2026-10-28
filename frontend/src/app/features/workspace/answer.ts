import { DecimalPipe } from '@angular/common';
import { Component, computed, inject, input, output } from '@angular/core';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MarkdownComponent } from 'ngx-markdown';

import { ChatStore } from '../../core/chat.store';
import { Turn } from '../../core/models';
import { Pipeline } from './pipeline';

/** One Q&A exchange. Citations [n] become chips that open the matching source. */
@Component({
  selector: 'app-answer',
  imports: [MarkdownComponent, Pipeline, MatTooltipModule, DecimalPipe],
  template: `
    <div class="question">
      <div class="bubble">{{ turn().question }}</div>
    </div>

    <article class="answer" [class.selected]="selected()" (click)="select.emit()">
      <div class="avatar">
        <span class="ms fill">{{ agentIcon() }}</span>
      </div>
      <div class="body">
        @if (turn().streaming || showPipeline()) {
          <app-pipeline [turn]="turn()" />
        }
        @if (turn().search_query && turn().search_query !== turn().question) {
          <div class="rewrite">
            <span class="ms sm">manage_search</span> searched for “{{ turn().search_query }}”
          </div>
        }
        @if (turn().tools.length) {
          <div class="tools">
            @for (t of turn().tools; track $index) {
              <span class="chip"
                ><span class="ms sm">build</span>{{ t.name }}({{ t.argument }})</span
              >
            }
          </div>
        }

        @if (turn().answer) {
          <markdown class="md" [data]="rendered()" (click)="onClick($event)" />
        } @else if (turn().streaming) {
          <div class="typing"><span></span><span></span><span></span></div>
        }
        @if (turn().error) {
          <div class="error"><span class="ms sm">error</span>{{ turn().error }}</div>
        }

        @if (!turn().streaming) {
          <footer>
            <span class="chip" matTooltip="Model that generated the answer">
              <span class="ms sm">smart_toy</span>{{ shortModel() }}
            </span>
            @if (turn().usage; as u) {
              <span class="chip" matTooltip="End-to-end graph time">
                <span class="ms sm">timer</span>{{ totalMs() / 1000 | number: '1.1-1' }}s
              </span>
              <span class="chip" matTooltip="Input / output tokens across all LLM calls">
                <span class="ms sm">token</span>{{ u.totals.input_tokens | number }} /
                {{ u.totals.output_tokens | number }}
              </span>
              <span class="chip" matTooltip="Estimated cost of this answer">
                \${{ u.totals.cost_usd | number: '1.4-4' }}
              </span>
            }
            @if (turn().evals?.faithfulness != null) {
              <span
                class="chip"
                [class]="scoreClass(turn().evals!.faithfulness!)"
                matTooltip="Judge faithfulness (grounded in sources)"
              >
                <span class="ms sm">verified</span
                >{{ turn().evals!.faithfulness! * 100 | number: '1.0-0' }}% grounded
              </span>
            }
            <span class="chip" [class]="guardClass()" [matTooltip]="guardTooltip()">
              <span class="ms sm">shield</span>{{ guardLabel() }}
            </span>
            <button class="btn ghost inspect" (click)="inspect.emit()">
              <span class="ms sm">data_exploration</span> Inspect
            </button>
          </footer>
        }
      </div>
    </article>
  `,
  styles: `
    :host {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .question {
      display: flex;
      justify-content: flex-end;
    }
    .bubble {
      max-width: 78%;
      background: var(--user-bubble);
      color: var(--user-ink);
      padding: 10px 14px;
      border-radius: 16px 16px 4px 16px;
      white-space: pre-wrap;
    }
    .answer {
      display: flex;
      gap: 12px;
      padding: 12px;
      margin: -12px;
      border-radius: 14px;
      border: 1px solid transparent;
      transition: border-color 0.15s;
      &.selected {
        border-color: var(--border);
        background: color-mix(in srgb, var(--surface) 60%, transparent);
      }
    }
    .avatar {
      flex: none;
      width: 30px;
      height: 30px;
      border-radius: 9px;
      background: var(--accent-soft);
      color: var(--accent);
      display: grid;
      place-items: center;
      .ms {
        font-size: 18px;
      }
    }
    .body {
      flex: 1;
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .rewrite {
      display: flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 12.5px;
      font-style: italic;
    }
    .tools {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      .chip {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
      }
    }
    footer {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
      padding-top: 2px;
    }
    .inspect {
      margin-left: auto;
      font-size: 12.5px;
      padding: 4px 8px;
      color: var(--accent);
    }
    .error {
      display: flex;
      align-items: center;
      gap: 6px;
      color: var(--block);
      background: var(--block-soft);
      padding: 8px 10px;
      border-radius: 8px;
    }
    .typing {
      display: flex;
      gap: 4px;
      padding: 6px 0;
      span {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--muted);
        animation: blink 1.2s infinite ease-in-out;
      }
      span:nth-child(2) {
        animation-delay: 0.15s;
      }
      span:nth-child(3) {
        animation-delay: 0.3s;
      }
    }
    @keyframes blink {
      0%,
      80%,
      100% {
        opacity: 0.25;
      }
      40% {
        opacity: 1;
      }
    }
  `,
})
export class Answer {
  private readonly store = inject(ChatStore);
  readonly turn = input.required<Turn>();
  readonly selected = input(false);
  readonly showPipeline = input(false);
  readonly select = output<void>();
  readonly inspect = output<void>();
  readonly cite = output<number>();

  /** [n] -> markdown link to #cite-n (rendered as a chip by the global .md styles). */
  protected readonly rendered = computed(() =>
    this.turn().answer.replace(/\[(\d{1,2})\](?!\()/g, '[$1](#cite-$1)'),
  );

  protected readonly agentIcon = computed(
    () =>
      this.store.config()?.agents.find((a) => a.name === this.turn().agent)?.icon ?? 'smart_toy',
  );
  protected readonly shortModel = computed(() => this.turn().model.split('/').pop() ?? '');
  protected readonly totalMs = computed(() =>
    Object.values(this.turn().usage?.timings ?? {}).reduce((a, b) => a + b, 0),
  );
  private readonly tripped = computed(() =>
    this.turn().guardrails.filter((g) => g.action !== 'pass'),
  );
  protected readonly guardClass = computed(() => {
    const t = this.tripped();
    if (t.some((g) => g.action === 'block')) return 'block';
    if (t.some((g) => g.action === 'flag')) return 'flag';
    if (t.some((g) => g.action === 'redact')) return 'redact';
    return 'pass';
  });
  protected readonly guardLabel = computed(() => {
    const t = this.tripped();
    return t.length
      ? `${t.length} guardrail${t.length > 1 ? 's' : ''}`
      : `${this.turn().guardrails.length} checks passed`;
  });
  protected readonly guardTooltip = computed(() =>
    this.tripped().length
      ? this.tripped()
          .map((g) => `${g.name}: ${g.action}`)
          .join('\n')
      : 'All guardrails passed',
  );

  protected scoreClass(score: number): string {
    return score >= 0.8 ? 'pass' : score >= 0.5 ? 'flag' : 'block';
  }

  protected onClick(event: MouseEvent): void {
    const link = (event.target as HTMLElement).closest('a');
    const href = link?.getAttribute('href') ?? '';
    if (href.startsWith('#cite-')) {
      event.preventDefault();
      event.stopPropagation();
      this.cite.emit(Number(href.slice(6)));
    }
  }
}
