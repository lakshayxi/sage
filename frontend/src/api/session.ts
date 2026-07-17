// Per-browser session token: opaque, server-issued (see
// api/routes/conversations.py's POST /conversations), persisted in
// localStorage so this visitor's conversations stay scoped to them across
// page loads without any account/login concept. Shared by conversations.ts
// (as an X-Session-Token header) and chat.ts (as a query param, since
// EventSource can't set headers -- see chat.ts).

const SESSION_TOKEN_KEY = 'sage_session_token'

export function getSessionToken(): string {
  return localStorage.getItem(SESSION_TOKEN_KEY) ?? ''
}

export function setSessionToken(token: string): void {
  localStorage.setItem(SESSION_TOKEN_KEY, token)
}

// Vite build-time env var, unset unless the deployment configures it (see
// deploy/huggingface/DEPLOY.md). Gates /chat, /conversations, /documents
// behind api/middleware.py's DemoKeyMiddleware.
export const DEMO_ACCESS_KEY: string | undefined = import.meta.env.VITE_DEMO_ACCESS_KEY

export function demoKeyHeaders(): HeadersInit {
  return DEMO_ACCESS_KEY ? { 'X-Demo-Key': DEMO_ACCESS_KEY } : {}
}
