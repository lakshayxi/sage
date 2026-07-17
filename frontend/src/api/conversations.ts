import type { ConversationDetailResponse, ConversationSummaryOut } from './types'
import { demoKeyHeaders, getSessionToken, setSessionToken } from './session'

function scopedHeaders(): HeadersInit {
  const token = getSessionToken()
  return {
    ...demoKeyHeaders(),
    ...(token ? { 'X-Session-Token': token } : {}),
  }
}

export async function listConversations(): Promise<ConversationSummaryOut[]> {
  // No session yet (nothing in localStorage) means this visitor has no
  // conversations -- skip the request rather than asking the backend, which
  // would 200 with an empty list anyway for a missing/unknown token.
  if (!getSessionToken()) return []

  const response = await fetch('/conversations', { headers: scopedHeaders() })
  if (!response.ok) throw new Error(`Failed to load conversations (${response.status})`)
  return response.json()
}

export async function getConversation(id: number): Promise<ConversationDetailResponse> {
  const response = await fetch(`/conversations/${id}`, { headers: scopedHeaders() })
  if (!response.ok) throw new Error(`Failed to load conversation (${response.status})`)
  return response.json()
}

export async function createConversation(title?: string): Promise<number> {
  const response = await fetch('/conversations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...scopedHeaders() },
    body: JSON.stringify({ title: title ?? null }),
  })
  if (!response.ok) throw new Error(`Failed to create conversation (${response.status})`)
  const data: { conversation_id: number; session_token: string } = await response.json()
  // The server reuses the token we sent (if any) or mints a fresh one on
  // this visitor's first-ever conversation -- persist whichever it returns.
  setSessionToken(data.session_token)
  return data.conversation_id
}
