import type { ChatResponse } from './types'
import { demoKeyHeaders, getSessionToken } from './session'

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

// POST /chat/stream via fetch() + a streamed ReadableStream, not the
// browser's native EventSource. EventSource can only ever GET and can't set
// custom request headers, which is why this used to carry the query,
// session token, and demo access key as URL query params (server access
// logs, browser history, and any Referer header a URL leaks through, all
// exposed them). fetch() lets this endpoint look like every other POST
// route instead: query/filters in a JSON body, session token and demo key
// as real headers (X-Session-Token / X-Demo-Key, matching conversations.ts
// and documents.ts) -- see api/routes/chat.py and api/middleware.py.
//
// The response body is still SSE-formatted text ("data: {...}\n\n",
// "event: done\ndata: {...}\n\n", "event: error\ndata: {...}\n\n"), just
// read and framed here instead of by the browser's built-in EventSource
// parser, so the incremental-delta / final-`done` / `error` contract with
// the backend is unchanged.
//
// Returns an AbortController the caller can use to cancel the in-flight
// request (e.g. on unmount) -- the EventSource this replaces exposed a
// similar `.close()`.
export function streamChat(params: StreamChatParams, handlers: StreamChatHandlers): AbortController {
  const controller = new AbortController()

  const body: Record<string, unknown> = { query: params.query }
  if (params.companies?.length) body.companies = params.companies
  if (params.fiscalYear) body.fiscal_year = params.fiscalYear
  if (params.docType) body.doc_type = params.docType
  if (params.sessionId != null) body.session_id = params.sessionId

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...demoKeyHeaders(),
  }
  if (params.sessionId != null) {
    const token = getSessionToken()
    if (token) headers['X-Session-Token'] = token
  }

  void runStream(controller, body, headers, handlers)

  return controller
}

async function runStream(
  controller: AbortController,
  body: Record<string, unknown>,
  headers: Record<string, string>,
  handlers: StreamChatHandlers,
): Promise<void> {
  // Whether a `done` or `error` frame has actually been dispatched to the
  // caller yet. The caller's Promise (useChatSession.ts) only ever resolves
  // from inside onDone/onError -- if the connection closes (cleanly or not)
  // without either firing, that Promise hangs forever and the UI stays
  // stuck in "streaming" state with no way to recover. EventSource, which
  // this replaced, didn't have this failure mode: the browser fires its own
  // error event automatically on an unexpected close. fetch()+ReadableStream
  // has no equivalent, so it's reproduced explicitly below.
  let sawTerminalEvent = false

  const dispatch = (frame: string): void => {
    if (dispatchFrame(frame, handlers)) sawTerminalEvent = true
  }

  try {
    const response = await fetch('/chat/stream', {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
    })

    if (!response.ok || !response.body) {
      handlers.onError(`Sage returned an error (${response.status}) while starting the answer.`)
      return
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      let boundary = buffer.indexOf('\n\n')
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary)
        buffer = buffer.slice(boundary + 2)
        dispatch(frame)
        boundary = buffer.indexOf('\n\n')
      }
    }

    // Flush any multi-byte UTF-8 sequence the decoder was still holding
    // onto, in case the stream ended mid-character.
    buffer += decoder.decode()

    // The connection closed with an unterminated frame still buffered --
    // e.g. the server wrote the `done` event's JSON but the connection cut
    // out before its trailing blank line arrived. Best-effort: try to
    // parse it as a final frame anyway rather than silently discarding a
    // `done`/`error` payload that mostly did arrive.
    if (!sawTerminalEvent && buffer.trim()) {
      try {
        dispatch(buffer)
      } catch {
        // Genuinely truncated (e.g. cut off mid-JSON) -- fall through to
        // the sawTerminalEvent check below, which reports the failure.
      }
    }

    if (!sawTerminalEvent) {
      handlers.onError('Lost connection to Sage while generating the answer.')
    }
  } catch {
    if (controller.signal.aborted) return
    if (!sawTerminalEvent) {
      handlers.onError('Lost connection to Sage while generating the answer.')
    }
  }
}

// Parses one blank-line-terminated SSE frame ("event: name\ndata: ...\n..."
// or a bare "data: ..." delta with no event line, which defaults to a
// delta like EventSource's unnamed "message" event did). Returns true if a
// terminal (`done`/`error`) event was dispatched.
function dispatchFrame(frame: string, handlers: StreamChatHandlers): boolean {
  let eventName = 'delta'
  const dataLines: string[] = []
  for (const line of frame.split('\n')) {
    if (line.startsWith('event: ')) {
      eventName = line.slice('event: '.length).trim()
    } else if (line.startsWith('data: ')) {
      dataLines.push(line.slice('data: '.length))
    }
  }
  if (dataLines.length === 0) return false
  const raw = dataLines.join('\n')

  if (eventName === 'done') {
    handlers.onDone(JSON.parse(raw) as ChatResponse)
    return true
  } else if (eventName === 'error') {
    const data = JSON.parse(raw) as { detail: string }
    handlers.onError(data.detail)
    return true
  } else {
    const data = JSON.parse(raw) as { delta: string }
    handlers.onDelta(data.delta)
    return false
  }
}
