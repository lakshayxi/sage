import { useState, type FormEvent } from 'react'

interface QueryBarProps {
  onSubmit: (query: string) => void
  disabled: boolean
  hasAskedBefore: boolean
}

export function QueryBar({ onSubmit, disabled, hasAskedBefore }: QueryBarProps) {
  const [value, setValue] = useState('')

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    const query = value.trim()
    if (!query || disabled) return
    onSubmit(query)
    setValue('')
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-2">
      <input
        type="text"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder={hasAskedBefore ? 'Ask another question…' : 'Ask about these filings…'}
        disabled={disabled}
        className="flex-1 rounded-lg border border-border bg-panel px-3.5 py-2.5 text-sm text-text placeholder:text-text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent disabled:opacity-60"
      />
      <button
        type="submit"
        disabled={disabled || !value.trim()}
        className="rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-on-accent transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
      >
        Ask
      </button>
    </form>
  )
}
