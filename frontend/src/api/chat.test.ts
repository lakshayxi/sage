import { beforeEach, describe, expect, it, vi } from 'vitest'
import { streamChat } from './chat'

// No frontend test suite existed before this file -- these are the first.
// Scoped narrowly to the /chat/stream parser, added because a real bug was
// found here during review: a stream that ends (server closes the
// connection) without ever delivering a complete `done`/`error` frame used
// to be silently dropped, leaving the caller's Promise (useChatSession.ts,
// which only resolves from inside onDone/onError) hanging forever with no
// way for the UI to recover. EventSource (what this replaced) didn't have
// this failure mode -- the browser fires its own error event automatically
// on an unexpected close; fetch()+ReadableStream has no equivalent, so
// chat.ts reproduces it explicitly (see runStream's `sawTerminalEvent`).

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder()
  let i = 0
  return new ReadableStream({
    pull(controller) {
      if (i < chunks.length) {
        controller.enqueue(encoder.encode(chunks[i]))
        i += 1
      } else {
        controller.close()
      }
    },
  })
}

function byteStreamOf(bytes: Uint8Array, chunkSize: number): ReadableStream<Uint8Array> {
  let offset = 0
  return new ReadableStream({
    pull(controller) {
      if (offset < bytes.length) {
        controller.enqueue(bytes.slice(offset, offset + chunkSize))
        offset += chunkSize
      } else {
        controller.close()
      }
    },
  })
}

function stubLocalStorage(): void {
  const store = new Map<string, string>()
  vi.stubGlobal('localStorage', {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => store.set(key, value),
    removeItem: (key: string) => store.delete(key),
  })
}

function mockFetchOnce(body: ReadableStream<Uint8Array> | null, status = 200): void {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      body,
    }),
  )
}

function collect() {
  const calls: Array<['delta' | 'done' | 'error', unknown]> = []
  return {
    calls,
    handlers: {
      onDelta: (t: string) => calls.push(['delta', t]),
      onDone: (r: unknown) => calls.push(['done', r]),
      onError: (m: string) => calls.push(['error', m]),
    },
  }
}

async function waitFor(predicate: () => boolean, timeoutMs = 1000): Promise<void> {
  const start = Date.now()
  while (!predicate()) {
    if (Date.now() - start > timeoutMs) throw new Error('waitFor timed out')
    await new Promise((r) => setTimeout(r, 0))
  }
}

describe('streamChat', () => {
  beforeEach(() => {
    stubLocalStorage()
  })

  it('does not put the query, session token, or demo key in the URL', async () => {
    mockFetchOnce(streamOf(['event: done\ndata: {"answer":"x","schema_version":1}\n\n']))
    const { handlers } = collect()

    streamChat({ query: 'secret query text', sessionId: 42 }, handlers)
    await waitFor(() => (fetch as unknown as { mock: { calls: unknown[] } }).mock.calls.length > 0)

    const [url, init] = (fetch as unknown as { mock: { calls: [string, RequestInit][] } }).mock
      .calls[0]
    expect(url).toBe('/chat/stream')
    expect(url).not.toContain('secret')
    expect(url).not.toContain('?')
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body as string)).toMatchObject({
      query: 'secret query text',
      session_id: 42,
    })
  })

  it('handles multiple SSE events arriving in a single network chunk', async () => {
    mockFetchOnce(
      streamOf([
        'data: {"delta": "A"}\n\ndata: {"delta": "B"}\n\n' +
          'event: done\ndata: {"answer": "AB", "schema_version": 1}\n\n',
      ]),
    )
    const { calls, handlers } = collect()

    streamChat({ query: 'q' }, handlers)
    await waitFor(() => calls.some((c) => c[0] === 'done'))

    expect(calls).toEqual([
      ['delta', 'A'],
      ['delta', 'B'],
      ['done', { answer: 'AB', schema_version: 1 }],
    ])
  })

  it('reassembles one SSE event split across many network chunks', async () => {
    mockFetchOnce(
      streamOf(['event: don', 'e\ndata: {"ans', 'wer": "hi", "sch', 'ema_version": 1}\n\n']),
    )
    const { calls, handlers } = collect()

    streamChat({ query: 'q' }, handlers)
    await waitFor(() => calls.some((c) => c[0] === 'done'))

    expect(calls).toEqual([['done', { answer: 'hi', schema_version: 1 }]])
  })

  it('reassembles UTF-8 multi-byte characters split across chunk boundaries', async () => {
    const text = 'NVIDIA’s margins grew — café 😀.'
    const deltaFrame = `data: ${JSON.stringify({ delta: text })}\n\n`
    const doneFrame = 'event: done\ndata: {"answer": "done", "schema_version": 1}\n\n'
    const bytes = new TextEncoder().encode(deltaFrame + doneFrame)
    // chunkSize=3 guarantees at least one multi-byte UTF-8 sequence (the
    // curly quote, em dash, "é", and the 4-byte emoji) gets split across
    // separate reads.
    mockFetchOnce(byteStreamOf(bytes, 3))
    const { calls, handlers } = collect()

    streamChat({ query: 'q' }, handlers)
    await waitFor(() => calls.some((c) => c[0] === 'done'))

    expect(calls).toEqual([
      ['delta', text],
      ['done', { answer: 'done', schema_version: 1 }],
    ])
  })

  it('recovers a final done event whose trailing blank line never arrived', async () => {
    // Regression test for the bug found during review: the connection
    // closes right after the done event's JSON, with no trailing "\n\n".
    mockFetchOnce(
      streamOf([
        'data: {"delta": "Margins declined."}\n\n',
        'event: done\ndata: {"answer": "Margins declined.", "schema_version": 1}',
      ]),
    )
    const { calls, handlers } = collect()

    streamChat({ query: 'q' }, handlers)
    await waitFor(() => calls.some((c) => c[0] === 'done'))

    expect(calls).toEqual([
      ['delta', 'Margins declined.'],
      ['done', { answer: 'Margins declined.', schema_version: 1 }],
    ])
  })

  it('surfaces an error instead of hanging when the stream ends with no terminal event', async () => {
    // Regression test: a connection that drops with only a partial delta
    // buffered (no done/error ever sent) must still call onError so the
    // caller's Promise (useChatSession.ts) doesn't hang forever.
    mockFetchOnce(streamOf(['data: {"delta": "partial answer"}\n\n']))
    const { calls, handlers } = collect()

    streamChat({ query: 'q' }, handlers)
    await waitFor(() => calls.some((c) => c[0] === 'error'))

    expect(calls).toEqual([
      ['delta', 'partial answer'],
      ['error', 'Lost connection to Sage while generating the answer.'],
    ])
  })

  it('does not double-fire onError after a clean done', async () => {
    mockFetchOnce(streamOf(['event: done\ndata: {"answer": "ok", "schema_version": 1}\n\n']))
    const { calls, handlers } = collect()

    streamChat({ query: 'q' }, handlers)
    await waitFor(() => calls.length > 0)
    await new Promise((r) => setTimeout(r, 20))

    expect(calls).toEqual([['done', { answer: 'ok', schema_version: 1 }]])
  })

  it('dispatches a server-sent error event', async () => {
    mockFetchOnce(streamOf(['event: error\ndata: {"detail": "boom"}\n\n']))
    const { calls, handlers } = collect()

    streamChat({ query: 'q' }, handlers)
    await waitFor(() => calls.some((c) => c[0] === 'error'))

    expect(calls).toEqual([['error', 'boom']])
  })

  it('reports an HTTP error that occurs before streaming starts', async () => {
    mockFetchOnce(null, 429)
    const { calls, handlers } = collect()

    streamChat({ query: 'q' }, handlers)
    await waitFor(() => calls.length > 0)

    expect(calls).toHaveLength(1)
    expect(calls[0][0]).toBe('error')
    expect(calls[0][1]).toContain('429')
  })

  it('does not call onError after the caller aborts the request', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((_url: string, init: RequestInit) => {
        return new Promise((_resolve, reject) => {
          init.signal?.addEventListener('abort', () => {
            const err = new Error('aborted')
            err.name = 'AbortError'
            reject(err)
          })
        })
      }),
    )
    const { calls, handlers } = collect()

    const controller = streamChat({ query: 'q' }, handlers)
    controller.abort()
    await new Promise((r) => setTimeout(r, 20))

    expect(calls).toEqual([])
  })
})
