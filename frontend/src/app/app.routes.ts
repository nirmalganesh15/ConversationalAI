import { Routes } from '@angular/router';

import { adminGuard, authGuard } from './core/guards';

export const routes: Routes = [
  { path: 'login', loadComponent: () => import('./features/login/login').then((m) => m.Login) },
  {
    path: 'evals',
    canActivate: [authGuard, adminGuard],
    loadComponent: () => import('./features/evals/evals-page').then((m) => m.EvalsPage),
  },
  {
    path: '',
    canActivate: [authGuard],
    loadComponent: () => import('./features/workspace/workspace').then((m) => m.Workspace),
  },
  { path: '**', redirectTo: '' },
];
