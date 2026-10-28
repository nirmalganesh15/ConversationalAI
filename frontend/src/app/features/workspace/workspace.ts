import { Component, ElementRef, computed, effect, inject, signal, viewChild } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatTooltipModule } from '@angular/material/tooltip';

import { ChatStore } from '../../core/chat.store';
import { Answer } from './answer';
import { Inspector, InspectorTab } from './inspector';
import { Knowledge } from './knowledge';
import { Sidebar } from './sidebar';

@Component({
  selector: 'app-workspace',
  imports: [FormsModule, MatTooltipModule, Sidebar, Answer, Inspector, Knowledge],
  templateUrl: './workspace.html',
  styleUrl: './workspace.scss',
})
export class Workspace {
  protected readonly store = inject(ChatStore);
  protected readonly inspectorOpen = signal(true);
  protected readonly knowledgeOpen = signal(false);
  protected readonly inspectorTab = signal<InspectorTab>('sources');
  protected readonly loadError = signal('');
  protected draft = '';
  private readonly thread = viewChild<ElementRef<HTMLElement>>('thread');

  protected readonly modelOptions = computed(() => this.store.providerInfo()?.models ?? []);

  constructor() {
    this.store.init().catch(() => this.loadError.set('Could not reach the API.'));
    // Keep the newest tokens in view while streaming.
    effect(() => {
      this.store.turns();
      const el = this.thread()?.nativeElement;
      if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 160) {
        queueMicrotask(() => el.scrollTo({ top: el.scrollHeight }));
      }
    });
  }

  protected send(text?: string): void {
    const question = (text ?? this.draft).trim();
    if (!question) return;
    this.draft = '';
    void this.store.ask(question);
    queueMicrotask(() => {
      const el = this.thread()?.nativeElement;
      el?.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
    });
  }

  protected onKey(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.send();
    }
  }

  protected openInspector(index: number, tab: InspectorTab = 'sources'): void {
    this.store.selectedTurn.set(index);
    this.inspectorTab.set(tab);
    this.inspectorOpen.set(true);
  }

  protected onCite(index: number, source: number): void {
    this.openInspector(index);
    this.store.highlightedSource.set(null);
    queueMicrotask(() => this.store.highlightedSource.set(source));
  }

  protected selectedIndex(): number {
    const selected = this.store.selectedTurn();
    return selected >= 0 ? selected : this.store.turns().length - 1;
  }
}
