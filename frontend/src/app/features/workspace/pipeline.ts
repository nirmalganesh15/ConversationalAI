import { Component, computed, input } from '@angular/core';

import { PIPELINE } from '../../core/chat.store';
import { Turn } from '../../core/models';

type StepState = 'done' | 'active' | 'pending' | 'skipped';

/** Live view of the LangGraph run: which node is executing right now. */
@Component({
  selector: 'app-pipeline',
  template: `
    @for (step of steps(); track step.node; let last = $last) {
      <div class="step" [class]="step.state">
        <span class="dot">
          @switch (step.state) {
            @case ('done') {
              <span class="ms sm">check</span>
            }
            @case ('active') {
              <span class="spinner"></span>
            }
            @case ('skipped') {
              <span class="ms sm">remove</span>
            }
          }
        </span>
        <span class="label">{{ step.label }}</span>
      </div>
      @if (!last) {
        <span class="bar" [class.filled]="step.state === 'done' || step.state === 'skipped'"></span>
      }
    }
  `,
  styles: `
    :host {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
      font-size: 11.5px;
      color: var(--muted);
    }
    .step {
      display: flex;
      align-items: center;
      gap: 5px;
    }
    .dot {
      width: 18px;
      height: 18px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      border: 1.5px solid var(--border-strong);
      background: var(--surface);
      .ms {
        font-size: 13px;
      }
    }
    .done .dot {
      background: var(--accent);
      border-color: var(--accent);
      color: var(--accent-ink);
    }
    .done .label {
      color: var(--text-2);
    }
    .active .dot {
      border-color: var(--accent);
    }
    .active .label {
      color: var(--accent);
      font-weight: 600;
    }
    .skipped {
      opacity: 0.45;
    }
    .bar {
      width: 14px;
      height: 1.5px;
      background: var(--border-strong);
      &.filled {
        background: var(--accent);
      }
    }
    .spinner {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      border: 2px solid var(--accent);
      border-right-color: transparent;
      animation: spin 0.7s linear infinite;
    }
    @keyframes spin {
      to {
        transform: rotate(360deg);
      }
    }
  `,
})
export class Pipeline {
  readonly turn = input.required<Turn>();

  protected readonly steps = computed(() => {
    const turn = this.turn();
    const done = new Set(turn.completedNodes ?? []);
    return PIPELINE.map(({ node, label }) => {
      let state: StepState = 'pending';
      if (turn.activeNode === node) state = 'active';
      else if (done.has(node)) state = 'done';
      else if (!turn.streaming) state = 'skipped';
      return { node, label, state };
    });
  });
}
