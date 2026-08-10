import { useCallback, useEffect, useRef, useState } from 'react'
import { createConversation, getConversation } from '../api/conversations'
import { streamChat } from '../api/chat'
import type { ChatResponse, CitationOut } from '../api/types'

export interface ChatTurn {
  id: string
  query: string
  status: 'streaming' | 'done' | 'error'
  answerText: string
  result: ChatResponse | null
  citations: CitationOut[]
  fromHistory: boolean
  errorMessage: string | null
}

export interface QueryFilters {
  companies: string[]
  fiscalYear?: string
  docType?: string
}

let turnCounter = 0
function nextTurnId(): string {
  turnCounter += 1
  return `turn-${turnCounter}`
}

export function useChatSession(onConversationCreated: () => void) {
  const [conversationId, setConversationId] = useState<number | null>(null)
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [isBusy, setIsBusy] = useState(false)
  const activeRequest = useRef<{
    controller: AbortController
    turnId: string
    resolve: () => void
  } | null>(null)

  const stopGenerating = useCallback(() => {
    const request = activeRequest.current
    if (!request) return

    activeRequest.current = null
    request.controller.abort()
    setTurns((current) =>
      current.map((turn) =>
        turn.id === request.turnId ? { ...turn, status: 'done', result: null } : turn,
      ),
    )
    setIsBusy(false)
    request.resolve()
  }, [])

  useEffect(
    () => () => {
      // A route/app teardown should not leave the fetch or the Promise
      // returned by ask() alive after this session hook is gone.
      const request = activeRequest.current
      if (!request) return
      activeRequest.current = null
      request.controller.abort()
      request.resolve()
    },
    [],
  )

  const startNewConversation = useCallback(() => {
    stopGenerating()
    setConversationId(null)
    setTurns([])
  }, [stopGenerating])

  const resumeConversation = useCallback(
    async (id: number) => {
      stopGenerating()
      const detail = await getConversation(id)
      const loaded: ChatTurn[] = []
      for (const message of detail.messages) {
        if (message.role === 'user') {
          loaded.push({
            id: `history-${message.id}`,
            query: message.content,
            status: 'done',
            answerText: '',
            result: null,
            citations: [],
            fromHistory: true,
            errorMessage: null,
          })
          continue
        }
        const last = loaded[loaded.length - 1]
        if (last) {
          last.answerText = message.content
          last.citations = (message.citations ?? []).map((citation) => ({
            ...citation,
            text: citation.text ?? 'Excerpt unavailable for this older saved answer.',
            fiscal_year: citation.fiscal_year ?? null,
            doc_type: citation.doc_type ?? null,
          }))
        }
      }
      setConversationId(id)
      setTurns(loaded)
    },
    [stopGenerating],
  )

  const ask = useCallback(
    async (query: string, filters: QueryFilters) => {
      let activeConversationId = conversationId
      if (activeConversationId == null) {
        activeConversationId = await createConversation(query.slice(0, 80))
        setConversationId(activeConversationId)
        onConversationCreated()
      }

      const turnId = nextTurnId()
      setTurns((current) => [
        ...current,
        {
          id: turnId,
          query,
          status: 'streaming',
          answerText: '',
          result: null,
          citations: [],
          fromHistory: false,
          errorMessage: null,
        },
      ])
      setIsBusy(true)

      const updateTurn = (patch: Partial<ChatTurn>) => {
        setTurns((current) => current.map((t) => (t.id === turnId ? { ...t, ...patch } : t)))
      }

      await new Promise<void>((resolve) => {
        let settled = false
        const settle = () => {
          if (settled) return
          settled = true
          resolve()
        }

        const controller = streamChat(
          {
            query,
            companies: filters.companies,
            fiscalYear: filters.fiscalYear,
            docType: filters.docType,
            sessionId: activeConversationId,
          },
          {
            onDelta: (text) => {
              if (controller.signal.aborted) return
              setTurns((current) =>
                current.map((t) => (t.id === turnId ? { ...t, answerText: t.answerText + text } : t)),
              )
            },
            onDone: (result) => {
              if (controller.signal.aborted) return
              // `result.answer` is the backend's authoritative full text --
              // prefer it over whatever partial text the accumulated deltas
              // produced, so a streaming edge case never leaves a
              // permanently-truncated answer on screen.
              updateTurn({
                status: 'done',
                result,
                citations: result.citations,
                answerText: result.answer,
              })
              if (activeRequest.current?.controller === controller) activeRequest.current = null
              setIsBusy(false)
              settle()
            },
            onError: (message) => {
              if (controller.signal.aborted) return
              updateTurn({ status: 'error', errorMessage: message })
              if (activeRequest.current?.controller === controller) activeRequest.current = null
              setIsBusy(false)
              settle()
            },
          },
        )
        activeRequest.current = { controller, turnId, resolve: settle }
      })
    },
    [conversationId, onConversationCreated],
  )

  return {
    conversationId,
    turns,
    isBusy,
    ask,
    stopGenerating,
    startNewConversation,
    resumeConversation,
  }
}
