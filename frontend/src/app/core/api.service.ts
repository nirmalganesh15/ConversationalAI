import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { AppConfig, ChatSession, IngestReport, SourceRow, Turn } from './models';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);

  config(): Observable<AppConfig> {
    return this.http.get<AppConfig>('/api/config');
  }

  sessions(): Observable<ChatSession[]> {
    return this.http.get<ChatSession[]>('/api/sessions');
  }

  session(id: string): Observable<{ session: ChatSession; turns: Turn[]; summary: string }> {
    return this.http.get<{ session: ChatSession; turns: Turn[]; summary: string }>(
      `/api/sessions/${id}`,
    );
  }

  deleteSession(id: string): Observable<void> {
    return this.http.delete<void>(`/api/sessions/${id}`);
  }

  sources(agent: string): Observable<SourceRow[]> {
    return this.http.get<SourceRow[]>(`/api/knowledge/${agent}/sources`);
  }

  upload(agent: string, files: File[], docRole?: string): Observable<IngestReport[]> {
    const form = new FormData();
    files.forEach((f) => form.append('files', f, f.name));
    if (docRole) form.append('doc_role', docRole);
    return this.http.post<IngestReport[]>(`/api/knowledge/${agent}/upload`, form);
  }

  ingestGithub(url: string): Observable<IngestReport> {
    return this.http.post<IngestReport>('/api/knowledge/code/github', { url });
  }

  deleteSource(agent: string, source: string): Observable<void> {
    return this.http.delete<void>(`/api/knowledge/${agent}/sources`, { params: { source } });
  }

  evalReports(): Observable<EvalReport[]> {
    return this.http.get<EvalReport[]>('/api/evals/reports');
  }
}

// Offline eval reports are loosely typed: they are rendered generically by the dashboard.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type EvalReport = Record<string, any>;
