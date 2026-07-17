import { useCallback, useState } from 'react'
import { createConversation, getConversation } from '../api/conversations'
import { streamChat } from '../api/chat'
import type { ChatResponse } from '../api/types'

export interface ChatTurn {
  id: string
  query: string
  status: 'streaming' | 'done' | 'error'
  answerText: string
  result: ChatResponse | null
  // Turns loaded from a resumed conversation's history render with static
  // (non-interactive) citation markers and never feed the right panel --
  // only the numbering/citation set of the latest turn asked in *this*
  // browser session is guaranteed to still be in scope. See CitationMarker.
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

  const startNewConversation = useCallback(() => {
    setConversationId(null)
    setTurns([])
  }, [])

  const resumeConversation = useCallback(async (id: number) => {
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
          fromHistory: true,
          errorMessage: null,
        })
        continue
      }
      const last = loaded[loaded.length - 1]
      if (last) last.answerText = message.content
    }
    setConversationId(id)
    setTurns(loaded)
  }, [])

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
          fromHistory: false,
          errorMessage: null,
        },
      ])
      setIsBusy(true)

      const updateTurn = (patch: Partial<ChatTurn>) => {
        setTurns((current) => current.map((t) => (t.id === turnId ? { ...t, ...patch } : t)))
      }

      await new Promise<void>((resolve) => {
        streamChat(
          {
            query,
            companies: filters.companies,
            fiscalYear: filters.fiscalYear,
            docType: filters.docType,
            sessionId: activeConversationId,
          },
          {
            onDelta: (text) => {
              setTurns((current) =>
                current.map((t) => (t.id === turnId ? { ...t, answerText: t.answerText + text } : t)),
              )
            },
            onDone: (result) => {
              // `result.answer` is the backend's authoritative full text --
              // prefer it over whatever partial text the accumulated deltas
              // produced, so a streaming edge case never leaves a
              // permanently-truncated answer on screen.
              updateTurn({ status: 'done', result, answerText: result.answer })
              setIsBusy(false)
              resolve()
            },
            onError: (message) => {
              updateTurn({ status: 'error', errorMessage: message })
              setIsBusy(false)
              resolve()
            },
          },
        )
      })
    },
    [conversationId, onConversationCreated],
  )

  return { conversationId, turns, isBusy, ask, startNewConversation, resumeConversation }
}
