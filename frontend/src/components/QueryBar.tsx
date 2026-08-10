import type { FormEvent } from 'react'

interface QueryBarProps {
  onSubmit: (query: string) => void
  onStop: () => void
  value: string
  onChange: (value: string) => void
  disabled: boolean
  hasAskedBefore: boolean
}

export function QueryBar({
  onSubmit,
  onStop,
  value,
  onChange,
  disabled,
  hasAskedBefore,
}: QueryBarProps) {
  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    const query = value.trim()
    if (!query || disabled) return
    onSubmit(query)
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-2">
      <input
        type="text"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        aria-label="Question about the selected filings"
        placeholder={hasAskedBefore ? 'Ask another question…' : 'Ask about these filings…'}
        disabled={disabled}
        className="flex-1 rounded-lg border border-border bg-panel px-3.5 py-2.5 text-sm text-text placeholder:text-text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent disabled:opacity-60"
      />
      {disabled ? (
        <button
          type="button"
          onClick={onStop}
          className="inline-flex items-center gap-2 rounded-lg border border-negative/60 px-3.5 py-2.5 text-sm font-medium text-negative transition-colors hover:bg-negative/10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-negative"
        >
          <span aria-hidden="true" className="h-2.5 w-2.5 rounded-[2px] bg-current" />
          <span className="hidden sm:inline">Stop generating</span>
          <span className="sm:hidden">Stop</span>
        </button>
      ) : (
        <button
          type="submit"
          disabled={!value.trim()}
          className="rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-on-accent transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
        >
          Ask
        </button>
      )}
    </form>
  )
}
