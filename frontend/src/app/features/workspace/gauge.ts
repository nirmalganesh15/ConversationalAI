import { Component, computed, input } from '@angular/core';

/** Ring gauge for a 0..1 metric; colour encodes the quality band. */
@Component({
  selector: 'app-gauge',
  template: `
    <svg viewBox="0 0 64 64" aria-hidden="true">
      <circle cx="32" cy="32" r="26" class="track" />
      @if (value() !== null) {
        <circle
          cx="32"
          cy="32"
          r="26"
          class="value"
          [class]="band()"
          [attr.stroke-dasharray]="dash()"
          transform="rotate(-90 32 32)"
        />
      }
      <text x="32" y="36" text-anchor="middle">{{ label() }}</text>
    </svg>
    <span class="name">{{ name() }}</span>
  `,
  styles: `
    :host {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
    }
    svg {
      width: 72px;
      height: 72px;
    }
    circle {
      fill: none;
      stroke-width: 6;
    }
    .track {
      stroke: var(--surface-3);
    }
    .value {
      stroke-linecap: round;
      transition: stroke-dasharray 0.5s ease;
      &.pass {
        stroke: var(--pass);
      }
      &.flag {
        stroke: var(--flag);
      }
      &.block {
        stroke: var(--block);
      }
    }
    text {
      font-size: 14px;
      font-weight: 700;
      fill: var(--text);
    }
    .name {
      font-size: 11.5px;
      color: var(--muted);
      text-align: center;
    }
  `,
})
export class Gauge {
  readonly value = input<number | null>(null);
  readonly name = input('');

  private readonly circumference = 2 * Math.PI * 26;
  protected readonly dash = computed(() => {
    const v = Math.max(0, Math.min(1, this.value() ?? 0));
    return `${v * this.circumference} ${this.circumference}`;
  });
  protected readonly label = computed(() =>
    this.value() === null ? '–' : `${Math.round((this.value() ?? 0) * 100)}`,
  );
  protected readonly band = computed(() => {
    const v = this.value() ?? 0;
    return v >= 0.8 ? 'pass' : v >= 0.5 ? 'flag' : 'block';
  });
}
