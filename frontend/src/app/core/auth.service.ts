import { HttpClient } from '@angular/common/http';
import { Injectable, computed, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { firstValueFrom } from 'rxjs';

import { User } from './models';

const STORAGE_KEY = 'atlas.auth';

interface Stored {
  token: string;
  user: User;
}

function readStored(): Stored | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Stored) : null;
  } catch {
    return null;
  }
}

/** Holds the JWT + user. Role checks here only shape the UI; the API enforces access. */
@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly http = inject(HttpClient);
  private readonly router = inject(Router);
  private readonly state = signal<Stored | null>(readStored());

  readonly user = computed(() => this.state()?.user ?? null);
  readonly token = computed(() => this.state()?.token ?? null);
  readonly isAuthenticated = computed(() => this.state() !== null);
  readonly isAdmin = computed(() => this.user()?.is_admin ?? false);

  async login(username: string, password: string): Promise<void> {
    const result = await firstValueFrom(
      this.http.post<Stored>('/api/auth/login', { username, password }),
    );
    this.state.set(result);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(result));
    } catch {
      /* private mode: session still works until reload */
    }
  }

  logout(): void {
    this.state.set(null);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* ignore */
    }
    void this.router.navigate(['/login']);
  }
}
