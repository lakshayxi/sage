import { useEffect, useRef } from 'react'
import type { ChatTurn } from '../hooks/useChatSession'
import { AnswerTurn } from './AnswerTurn'
import { QueryBar } from './QueryBar'

interface MainPanelProps {
  turns: ChatTurn[]
  onAsk: (query: string) => void
  isBusy: boolean
}

export function MainPanel({ turns, onAsk, isBusy }: MainPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [turns])

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollRef} className="thin-scroll flex-1 overflow-y-auto px-6 py-6">
        {turns.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <p className="font-display text-2xl text-text">What do you want to know?</p>
            <p className="mt-2 max-w-sm text-sm text-text-muted">
              Ask about the ingested Apple, Microsoft, and NVIDIA 10-Ks -- every claim comes back
              cited to a page in the source filing.
            </p>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl space-y-8">
            {turns.map((turn) => (
              <AnswerTurn key={turn.id} turn={turn} />
            ))}
          </div>
        )}
      </div>
      <div className="border-t border-border px-6 py-4">
        <div className="mx-auto max-w-3xl">
          <QueryBar onSubmit={onAsk} disabled={isBusy} hasAskedBefore={turns.length > 0} />
        </div>
      </div>
    </div>
  )
}
