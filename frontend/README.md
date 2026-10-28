# Atlas frontend

Angular 21 SPA (standalone components, signals, zoneless). See the root README.

- `npx ng serve` — dev server on :4200, proxies `/api` to `localhost:8000` (`proxy.conf.json`)
- `npx ng test --watch=false` — unit tests (vitest)
- `npx ng build` — production build to `dist/browser` (served by FastAPI in the container)
