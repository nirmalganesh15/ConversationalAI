import { TestBed } from '@angular/core/testing';

import { Turn } from '../../core/models';
import { Pipeline } from './pipeline';

function turn(patch: Partial<Turn>): Turn {
  return {
    question: 'q',
    answer: '',
    agent: 'docs',
    provider: 'fake',
    model: 'm',
    sources: [],
    tools: [],
    guardrails: [],
    evals: null,
    usage: null,
    search_query: '',
    ...patch,
  };
}

function states(t: Turn): string[] {
  const fixture = TestBed.createComponent(Pipeline);
  fixture.componentRef.setInput('turn', t);
  fixture.detectChanges();
  return Array.from(fixture.nativeElement.querySelectorAll('.step') as NodeListOf<HTMLElement>).map(
    (el) => el.className.replace('step', '').trim(),
  );
}

describe('Pipeline', () => {
  it('marks completed, active and pending nodes while streaming', () => {
    expect(
      states(
        turn({ streaming: true, activeNode: 'retrieve', completedNodes: ['input_guard', 'plan'] }),
      ),
    ).toEqual(['done', 'done', 'pending', 'active', 'pending', 'pending', 'pending']);
  });

  it('marks nodes that never ran as skipped once the turn finishes', () => {
    // A blocked request stops after the input guard.
    expect(states(turn({ streaming: false, completedNodes: ['input_guard'] }))).toEqual([
      'done',
      'skipped',
      'skipped',
      'skipped',
      'skipped',
      'skipped',
      'skipped',
    ]);
  });
});
