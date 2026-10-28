import { signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { Router, UrlTree, provideRouter } from '@angular/router';

import { AuthService } from './auth.service';
import { adminGuard, authGuard } from './guards';

function run(guard: typeof authGuard, auth: Partial<AuthService>) {
  TestBed.configureTestingModule({
    providers: [provideRouter([]), { provide: AuthService, useValue: auth }],
  });
  return TestBed.runInInjectionContext(() => guard({} as never, {} as never));
}

describe('route guards (UI-only; the API enforces RBAC independently)', () => {
  it('sends anonymous users to /login', () => {
    const result = run(authGuard, { isAuthenticated: signal(false) });
    expect(TestBed.inject(Router).serializeUrl(result as UrlTree)).toBe('/login');
  });

  it('lets authenticated users through', () => {
    expect(run(authGuard, { isAuthenticated: signal(true) })).toBe(true);
  });

  it('keeps staff away from the admin eval dashboard', () => {
    const result = run(adminGuard, { isAdmin: signal(false) });
    expect(TestBed.inject(Router).serializeUrl(result as UrlTree)).toBe('/');
  });
});
