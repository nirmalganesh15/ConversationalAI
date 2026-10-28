import { HttpErrorResponse } from '@angular/common/http';
import { Component, computed, effect, inject, output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatTooltipModule } from '@angular/material/tooltip';
import { firstValueFrom } from 'rxjs';

import { ApiService } from '../../core/api.service';
import { AuthService } from '../../core/auth.service';
import { ChatStore } from '../../core/chat.store';
import { SourceRow } from '../../core/models';

/** Slide-over panel to see what an agent knows and add to it. */
@Component({
  selector: 'app-knowledge',
  imports: [FormsModule, MatTooltipModule],
  template: `
    <div class="backdrop" (click)="close.emit()"></div>
    <aside class="drawer">
      <header>
        <div>
          <div class="eyebrow">Knowledge base</div>
          <h3>{{ store.agentInfo()?.title }}</h3>
        </div>
        <button class="btn ghost icon" (click)="close.emit()" aria-label="Close">
          <span class="ms">close</span>
        </button>
      </header>

      <section class="upload">
        @if (agent() === 'career') {
          <div class="roles">
            <label
              ><input type="radio" name="role" value="resume" [(ngModel)]="docRole" /> Resume</label
            >
            <label
              ><input type="radio" name="role" value="jd" [(ngModel)]="docRole" /> Job
              description</label
            >
          </div>
        }
        <label
          class="drop"
          [class.over]="dragOver()"
          (dragover)="onDrag($event, true)"
          (dragleave)="onDrag($event, false)"
          (drop)="onDrop($event)"
        >
          <span class="ms">upload_file</span>
          <span><strong>Drop files</strong> or click to upload</span>
          <small>{{ accepts() }}</small>
          <input type="file" multiple [accept]="accepts()" (change)="onPick($event)" hidden />
        </label>
        @if (agent() === 'code') {
          <div class="github">
            <input
              class="field"
              placeholder="https://github.com/owner/repo"
              [(ngModel)]="repoUrl"
              [disabled]="!auth.isAdmin()"
            />
            <button
              class="btn primary"
              (click)="ingestGithub()"
              [disabled]="!auth.isAdmin() || !repoUrl || busy()"
              [matTooltip]="
                auth.isAdmin() ? 'Index a public repository' : 'Admin only (least privilege)'
              "
            >
              Index
            </button>
          </div>
        }
        @if (busy()) {
          <div class="status">
            <span class="spinner"></span> Indexing… chunking, embedding, BM25
          </div>
        }
        @if (message()) {
          <div class="status" [class.err]="isError()">{{ message() }}</div>
        }
      </section>

      <div class="eyebrow">Sources ({{ rows().length }})</div>
      <div class="list scroll">
        @for (row of rows(); track row.source) {
          <div class="row">
            <span class="ms sm">{{
              row.doc_type === 'code'
                ? 'code'
                : row.doc_type === 'pdf'
                  ? 'picture_as_pdf'
                  : 'article'
            }}</span>
            <div class="meta">
              <span class="name">{{ row.title }}</span>
              <small
                >{{ row.chunks }} chunks ·
                {{ row.shared ? 'shared corpus' : 'uploaded by you' }}</small
              >
            </div>
            @if (row.deletable) {
              <button class="btn ghost icon" (click)="remove(row)" matTooltip="Remove from index">
                <span class="ms sm">delete</span>
              </button>
            }
          </div>
        } @empty {
          <p class="muted">Nothing indexed yet.</p>
        }
      </div>
    </aside>
  `,
  styles: `
    :host {
      position: fixed;
      inset: 0;
      z-index: 50;
    }
    .backdrop {
      position: absolute;
      inset: 0;
      background: rgb(0 0 0 / 0.25);
      animation: fade 0.2s;
    }
    .drawer {
      position: absolute;
      top: 0;
      right: 0;
      bottom: 0;
      width: min(440px, 100%);
      background: var(--surface);
      box-shadow: var(--shadow-lg);
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding: 18px;
      animation: slide 0.22s ease-out;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      h3 {
        margin: 2px 0 0;
        font-size: 18px;
      }
    }
    .drop {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      padding: 22px;
      border: 1.5px dashed var(--border-strong);
      border-radius: 14px;
      cursor: pointer;
      text-align: center;
      color: var(--text-2);
      .ms {
        font-size: 30px;
        color: var(--accent);
      }
      small {
        color: var(--muted);
      }
      &.over,
      &:hover {
        border-color: var(--accent);
        background: var(--accent-soft);
      }
    }
    .upload {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .roles {
      display: flex;
      gap: 16px;
      label {
        display: flex;
        gap: 6px;
        align-items: center;
      }
    }
    .github {
      display: flex;
      gap: 8px;
    }
    .status {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12.5px;
      color: var(--pass);
      &.err {
        color: var(--block);
      }
    }
    .list {
      flex: 1;
      min-height: 0;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .row {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 10px;
      border: 1px solid var(--border);
      border-radius: 10px;
      > .ms {
        color: var(--muted);
      }
    }
    .meta {
      flex: 1;
      min-width: 0;
      display: flex;
      flex-direction: column;
      .name {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-weight: 500;
      }
      small {
        color: var(--muted);
      }
    }
    .muted {
      color: var(--muted);
    }
    .spinner {
      width: 12px;
      height: 12px;
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
    @keyframes slide {
      from {
        transform: translateX(30px);
        opacity: 0;
      }
    }
    @keyframes fade {
      from {
        opacity: 0;
      }
    }
  `,
})
export class Knowledge {
  private readonly api = inject(ApiService);
  protected readonly store = inject(ChatStore);
  protected readonly auth = inject(AuthService);
  readonly close = output<void>();

  protected readonly agent = this.store.agent;
  protected readonly rows = signal<SourceRow[]>([]);
  protected readonly busy = signal(false);
  protected readonly message = signal('');
  protected readonly isError = signal(false);
  protected readonly dragOver = signal(false);
  protected docRole = 'resume';
  protected repoUrl = '';

  protected readonly accepts = computed(() =>
    (this.store.agentInfo()?.accepts ?? []).filter((a) => a.startsWith('.')).join(', '),
  );

  constructor() {
    effect(() => {
      void this.load(this.agent());
    });
  }

  private async load(agent: string): Promise<void> {
    this.rows.set(await firstValueFrom(this.api.sources(agent)));
  }

  protected onDrag(event: DragEvent, over: boolean): void {
    event.preventDefault();
    this.dragOver.set(over);
  }

  protected onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragOver.set(false);
    void this.upload(Array.from(event.dataTransfer?.files ?? []));
  }

  protected onPick(event: Event): void {
    const input = event.target as HTMLInputElement;
    void this.upload(Array.from(input.files ?? []));
    input.value = '';
  }

  private async upload(files: File[]): Promise<void> {
    if (!files.length) return;
    await this.run(async () => {
      const role = this.agent() === 'career' ? this.docRole : undefined;
      const reports = await firstValueFrom(this.api.upload(this.agent(), files, role));
      const chunks = reports.reduce((sum, r) => sum + r.chunks, 0);
      return `Indexed ${files.length} file(s) into ${chunks} chunks.`;
    });
  }

  protected async ingestGithub(): Promise<void> {
    await this.run(async () => {
      const report = await firstValueFrom(this.api.ingestGithub(this.repoUrl));
      this.repoUrl = '';
      return `Indexed ${report.documents} files into ${report.chunks} chunks.`;
    });
  }

  protected async remove(row: SourceRow): Promise<void> {
    await this.run(async () => {
      await firstValueFrom(this.api.deleteSource(this.agent(), row.source));
      return `Removed ${row.title}.`;
    });
  }

  private async run(action: () => Promise<string>): Promise<void> {
    this.busy.set(true);
    this.message.set('');
    try {
      this.message.set(await action());
      this.isError.set(false);
      await this.load(this.agent());
    } catch (e) {
      this.isError.set(true);
      this.message.set((e as HttpErrorResponse).error?.detail ?? 'Request failed.');
    } finally {
      this.busy.set(false);
    }
  }
}
