import type { ChatTurn } from '../hooks/useChatSession'
import { AnswerMarkdown } from '../lib/renderAnswerText'

export function AnswerTurn({ turn }: { turn: ChatTurn }) {
  const interactive = !turn.fromHistory

  return (
    <article className="space-y-3">
      <p className="font-mono text-sm text-text-muted">
        <span className="text-accent">Q</span> {turn.query}
      </p>

      {turn.status === 'error' ? (
        <p className="text-sm text-negative">{turn.errorMessage ?? 'Something went wrong.'}</p>
      ) : (
        <div className="space-y-3 text-[0.95rem] text-text">
          <AnswerMarkdown text={turn.answerText} interactive={interactive} />
          {turn.status === 'streaming' && (
            <span
              aria-hidden="true"
              className="inline-block h-4 w-1.5 translate-y-0.5 animate-pulse bg-accent motion-reduce:animate-none"
            />
          )}
        </div>
      )}
    </article>
  )
}
