import type { ChatResponse } from '../api/types'

function Stat({ label, value, valueClassName = '' }: { label: string; value: string; valueClassName?: string }) {
  return (
    <div>
      <p className="text-xs text-text-muted">{label}</p>
      <p className={`font-mono text-sm text-text ${valueClassName}`}>{value}</p>
    </div>
  )
}

export function MetadataBar({ result }: { result: ChatResponse }) {
  return (
    <div className="grid grid-cols-2 gap-3 rounded-lg border border-border bg-panel px-3 py-3">
      <Stat label="Model" value={result.model} />
      <Stat
        label="Cache"
        value={result.cache_hit ? 'hit' : 'miss'}
        valueClassName={result.cache_hit ? 'text-positive' : ''}
      />
      <Stat label="Latency" value={`${Math.round(result.latency_ms.total_ms)} ms`} />
      <Stat label="Tokens" value={String(result.tokens.total_tokens)} />
    </div>
  )
}
