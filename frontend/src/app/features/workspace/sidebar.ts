import { Component, computed, inject } from '@angular/core';
import { MatTooltipModule } from '@angular/material/tooltip';
import { RouterLink } from '@angular/router';

import { AuthService } from '../../core/auth.service';
import { ChatStore } from '../../core/chat.store';
import { ChatSession } from '../../core/models';
import { ThemeService } from '../../core/theme.service';

@Component({
  selector: 'app-sidebar',
  imports: [MatTooltipModule, RouterLink],
  template: `
    <div class="brand">
      <div class="logo">A</div>
      <span>Atlas</span>
      <span class="chip accent env">{{
        store.config()?.status?.seed === 'done' ? 'ready' : 'indexing…'
      }}</span>
    </div>

    <button class="btn primary new" (click)="store.newChat()">
      <span class="ms">add</span> New chat
    </button>

    <div class="eyebrow section">Agents</div>
    <nav class="agents">
      @for (a of store.config()?.agents ?? []; track a.name) {
        <button
          class="agent"
          [class.active]="store.agent() === a.name"
          (click)="store.newChat(a.name)"
          [matTooltip]="a.description"
          matTooltipPosition="right"
        >
          <span class="ms" [class.fill]="store.agent() === a.name">{{ a.icon }}</span>
          <span>{{ a.title }}</span>
        </button>
      }
    </nav>

    <div class="eyebrow section">History</div>
    <div class="sessions scroll">
      @for (s of store.sessions(); track s.id) {
        <div class="session" [class.active]="store.sessionId() === s.id" (click)="open(s)">
          <span class="ms sm">{{ iconFor(s.agent) }}</span>
          <span class="title">{{ s.title }}</span>
          <button class="btn ghost icon del" (click)="remove($event, s)" aria-label="Delete chat">
            <span class="ms sm">delete</span>
          </button>
        </div>
      } @empty {
        <p class="empty">No conversations yet.</p>
      }
    </div>

    <div class="footer">
      @if (auth.isAdmin()) {
        <a class="btn ghost evals" routerLink="/evals"
          ><span class="ms">monitoring</span> Eval reports</a
        >
      }
      <div class="user">
        <div class="avatar">{{ initial() }}</div>
        <div class="who">
          <strong>{{ auth.user()?.username }}</strong>
          <small>{{ auth.user()?.role }} · {{ store.config()?.agents?.length }} agents</small>
        </div>
        <button class="btn ghost icon" (click)="theme.toggle()" matTooltip="Toggle theme">
          <span class="ms">{{ theme.theme() === 'light' ? 'dark_mode' : 'light_mode' }}</span>
        </button>
        <button class="btn ghost icon" (click)="auth.logout()" matTooltip="Sign out">
          <span class="ms">logout</span>
        </button>
      </div>
    </div>
  `,
  styles: `
    :host {
      display: flex;
      flex-direction: column;
      height: 100%;
      padding: 16px 12px 12px;
      gap: 6px;
      background: var(--surface);
      border-right: 1px solid var(--border);
      min-height: 0;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 9px;
      font-weight: 700;
      font-size: 16px;
      padding: 0 6px 10px;
    }
    .logo {
      width: 28px;
      height: 28px;
      border-radius: 8px;
      background: var(--accent);
      color: var(--accent-ink);
      display: grid;
      place-items: center;
      font-weight: 800;
      font-size: 14px;
    }
    .env {
      margin-left: auto;
    }
    .new {
      justify-content: center;
      margin-bottom: 8px;
    }
    .section {
      padding: 10px 8px 4px;
    }
    .agents {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .agent {
      display: flex;
      align-items: center;
      gap: 10px;
      border: 0;
      background: transparent;
      padding: 8px 10px;
      border-radius: 9px;
      text-align: left;
      color: var(--text-2);
      font-weight: 500;
      &:hover {
        background: var(--surface-2);
      }
      &.active {
        background: var(--accent-soft);
        color: var(--accent);
      }
    }
    .sessions {
      flex: 1;
      min-height: 0;
      display: flex;
      flex-direction: column;
      gap: 1px;
    }
    .session {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 6px 6px 10px;
      border-radius: 8px;
      cursor: pointer;
      color: var(--text-2);
      .title {
        flex: 1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 13px;
      }
      .del {
        opacity: 0;
      }
      &:hover {
        background: var(--surface-2);
        .del {
          opacity: 0.7;
        }
      }
      &.active {
        background: var(--surface-2);
        color: var(--text);
      }
    }
    .empty {
      color: var(--muted);
      font-size: 12.5px;
      padding: 4px 10px;
    }
    .footer {
      border-top: 1px solid var(--border);
      padding-top: 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .evals {
      text-decoration: none;
      color: var(--text-2);
    }
    .user {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 4px;
    }
    .avatar {
      width: 30px;
      height: 30px;
      border-radius: 50%;
      background: var(--surface-3);
      display: grid;
      place-items: center;
      font-weight: 700;
      text-transform: uppercase;
    }
    .who {
      flex: 1;
      display: flex;
      flex-direction: column;
      line-height: 1.2;
      small {
        color: var(--muted);
      }
    }
  `,
})
export class Sidebar {
  protected readonly store = inject(ChatStore);
  protected readonly auth = inject(AuthService);
  protected readonly theme = inject(ThemeService);
  protected readonly initial = computed(() => this.auth.user()?.username.charAt(0) ?? '?');

  protected iconFor(agent: string): string {
    return this.store.config()?.agents.find((a) => a.name === agent)?.icon ?? 'chat';
  }

  protected open(session: ChatSession): void {
    void this.store.openSession(session);
  }

  protected remove(event: Event, session: ChatSession): void {
    event.stopPropagation();
    void this.store.deleteSession(session.id);
  }
}
