import { HttpErrorResponse } from '@angular/common/http';
import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

import { AuthService } from '../../core/auth.service';
import { ThemeService } from '../../core/theme.service';

@Component({
  selector: 'app-login',
  imports: [FormsModule],
  templateUrl: './login.html',
  styleUrl: './login.scss',
})
export class Login {
  private readonly auth = inject(AuthService);
  private readonly router = inject(Router);
  protected readonly theme = inject(ThemeService);

  protected username = '';
  protected password = '';
  protected readonly error = signal('');
  protected readonly loading = signal(false);

  protected readonly agents = [
    {
      icon: 'description',
      title: 'Clinical Docs',
      text: 'FDA labels, guidelines, trials',
      roles: 'admin · staff',
    },
    {
      icon: 'code',
      title: 'Code Docs',
      text: 'Ask a codebase how it works',
      roles: 'admin · staff',
    },
    {
      icon: 'groups',
      title: 'Meeting Intelligence',
      text: 'Decisions & action items',
      roles: 'admin',
    },
    {
      icon: 'work',
      title: 'Career Intelligence',
      text: 'Resume vs. job fit & gaps',
      roles: 'admin',
    },
  ];

  async submit(): Promise<void> {
    if (!this.username || !this.password) return;
    this.loading.set(true);
    this.error.set('');
    try {
      await this.auth.login(this.username, this.password);
      await this.router.navigate(['/']);
    } catch (e) {
      const status = (e as HttpErrorResponse).status;
      this.error.set(status === 401 ? 'Invalid username or password.' : 'Server unavailable.');
    } finally {
      this.loading.set(false);
    }
  }
}
