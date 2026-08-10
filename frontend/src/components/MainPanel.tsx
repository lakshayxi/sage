import { useEffect, useRef, useState } from 'react'
import type { ChatTurn } from '../hooks/useChatSession'
import { AnswerTurn } from './AnswerTurn'
import { QueryBar } from './QueryBar'

interface MainPanelProps {
  turns: ChatTurn[]
  onAsk: (query: string) => void
  onStop: () => void
  isBusy: boolean
}

const SAMPLE_QUESTIONS = [
  {
    label: 'Apple · performance',
    question: 'What were Apple’s total net sales and net income in FY2025?',
  },
  {
    label: 'Microsoft · investment',
    question: 'How much did Microsoft spend on research and development in FY2025?',
  },
  {
    label: 'Three-company comparison',
    question: 'Compare revenue across Apple FY2025, Microsoft FY2025, and NVIDIA FY2026.',
    instruction: 'Select Apple, Microsoft, and NVIDIA in Companies before asking.',
  },
] as const

export function MainPanel({ turns, onAsk, onStop, isBusy }: MainPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [draft, setDraft] = useState('')

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [turns])

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollRef} className="thin-scroll flex-1 overflow-y-auto px-6 py-6">
        {turns.length === 0 ? (
          <div className="mx-auto flex min-h-full max-w-3xl flex-col justify-center py-8">
            <div className="text-center">
              <p className="font-mono text-[0.68rem] uppercase tracking-[0.2em] text-accent">
                Research the filings
              </p>
              <p className="mt-2 font-display text-2xl text-text">What do you want to know?</p>
              <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-text-muted">
                Ask about the ingested Apple, Microsoft, and NVIDIA 10-Ks -- every claim comes back
                cited to a page in the source filing.
              </p>
            </div>

            <div className="mt-8 border-y border-border" aria-label="Sample questions">
              {SAMPLE_QUESTIONS.map((sample) => (
                <button
                  key={sample.question}
                  type="button"
                  onClick={() => setDraft(sample.question)}
                  className="group flex w-full flex-col gap-1 border-b border-border px-3 py-3 text-left transition-colors last:border-b-0 hover:bg-panel focus-visible:z-10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent sm:flex-row sm:items-start sm:gap-5"
                >
                  <span className="w-44 shrink-0 font-mono text-[0.68rem] uppercase tracking-wider text-text-muted group-hover:text-accent">
                    {sample.label}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm leading-snug text-text">{sample.question}</span>
                    {'instruction' in sample && (
                      <span className="mt-1 block text-xs leading-relaxed text-accent">
                        {sample.instruction}
                      </span>
                    )}
                  </span>
                  <span aria-hidden="true" className="hidden font-mono text-text-muted sm:block">
                    →
                  </span>
                </button>
              ))}
            </div>
            <p className="mt-3 text-center font-mono text-[0.68rem] text-text-muted">
              Choose a prompt to place it in the question bar.
            </p>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl space-y-8">
            {turns.map((turn, index) => (
              <AnswerTurn
                key={turn.id}
                turn={turn}
                interactive={index === turns.length - 1 && turn.citations.length > 0}
              />
            ))}
          </div>
        )}
      </div>
      <div className="border-t border-border px-6 py-4">
        <div className="mx-auto max-w-3xl">
          <QueryBar
            value={draft}
            onChange={setDraft}
            onSubmit={(query) => {
              onAsk(query)
              setDraft('')
            }}
            onStop={onStop}
            disabled={isBusy}
            hasAskedBefore={turns.length > 0}
          />
        </div>
      </div>
    </div>
  )
}
