// Per-browser session token: opaque, server-issued (see
// api/routes/conversations.py's POST /conversations), persisted in
// localStorage so this visitor's conversations stay scoped to them across
// page loads without any account/login concept. Sent as an X-Session-Token
// header by every caller (conversations.ts, documents.ts, and chat.ts's
// fetch()-based streaming) -- never as a URL query param, which would leak
// it into server access logs, browser history, and any Referer header.

const SESSION_TOKEN_KEY = 'sage_session_token'

export function getSessionToken(): string {
  return localStorage.getItem(SESSION_TOKEN_KEY) ?? ''
}

export function setSessionToken(token: string): void {
  localStorage.setItem(SESSION_TOKEN_KEY, token)
}

// Vite build-time env var, unset unless the deployment configures it (see
// deploy/huggingface/DEPLOY.md). Gates /chat, /conversations, /documents
// behind api/middleware.py's DemoKeyMiddleware -- NOT a real secret: Vite
// inlines this into the public JS bundle, so anyone can read it out of the
// deployed site's own source. It's a casual-access deterrent only; see
// api/middleware.py's module docstring for what actually protects the
// deployment (CHAT_RATE_LIMIT).
export const DEMO_ACCESS_KEY: string | undefined = import.meta.env.VITE_DEMO_ACCESS_KEY

export function demoKeyHeaders(): Record<string, string> {
  return DEMO_ACCESS_KEY ? { 'X-Demo-Key': DEMO_ACCESS_KEY } : {}
}
