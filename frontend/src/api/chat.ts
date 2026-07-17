import type { ChatResponse } from './types'
import { DEMO_ACCESS_KEY, getSessionToken } from './session'

export interface StreamChatParams {
  query: string
  companies?: string[]
  fiscalYear?: string
  docType?: string
  sessionId?: number | null
}

export interface StreamChatHandlers {
  onDelta: (text: string) => void
  onDone: (result: ChatResponse) => void
  onError: (message: string) => void
}

// GET /chat/stream via EventSource: unnamed "message" events carry
// {"delta": "..."} as text arrives, "done" carries the final ChatResponse,
// "error" carries {"detail": "..."}. EventSource also fires its own
// connection-level "error" (no .data) when the stream closes or a request
// fails outright -- both land on the same listener below, distinguished by
// whether event.data is set.
//
// EventSource can't set custom headers, so both the demo access key and the
// session token (needed to resume a conversation) go as query params here
// instead of the X-Demo-Key/X-Session-Token headers every other endpoint
// uses -- see api/middleware.py and api/routes/chat.py.
export function streamChat(params: StreamChatParams, handlers: StreamChatHandlers): EventSource {
  const url = new URL('/chat/stream', window.location.origin)
  url.searchParams.set('query', params.query)
  for (const company of params.companies ?? []) {
    url.searchParams.append('companies', company)
  }
  if (params.fiscalYear) url.searchParams.set('fiscal_year', params.fiscalYear)
  if (params.docType) url.searchParams.set('doc_type', params.docType)
  if (params.sessionId != null) {
    url.searchParams.set('session_id', String(params.sessionId))
    const token = getSessionToken()
    if (token) url.searchParams.set('session_token', token)
  }
  if (DEMO_ACCESS_KEY) url.searchParams.set('key', DEMO_ACCESS_KEY)

  const source = new EventSource(url)

  source.onmessage = (event) => {
    const data: { delta: string } = JSON.parse(event.data)
    handlers.onDelta(data.delta)
  }

  source.addEventListener('done', (event) => {
    const data: ChatResponse = JSON.parse((event as MessageEvent).data)
    handlers.onDone(data)
    source.close()
  })

  source.addEventListener('error', (event) => {
    const messageEvent = event as MessageEvent
    if (messageEvent.data) {
      const data: { detail: string } = JSON.parse(messageEvent.data)
      handlers.onError(data.detail)
    } else {
      handlers.onError('Lost connection to Sage while generating the answer.')
    }
    source.close()
  })

  return source
}
