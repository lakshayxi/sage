import { useCitationHighlight } from '../context/CitationHighlightContext'

interface CitationMarkerProps {
  n: number
  interactive: boolean
}

// The signature interaction: a compact gold-outlined mono numeral inline in
// the answer text, bidirectionally linked to its citation card in the right
// panel via CitationHighlightContext. Only the newest visible answer is
// interactive because citation numbering restarts per turn; older markers
// keep the same visual treatment without targeting the newest card set.
export function CitationMarker({ n, interactive }: CitationMarkerProps) {
  const { activeCitation, setActiveCitation, scrollToCard } = useCitationHighlight()
  const isActive = interactive && activeCitation === n

  const className = `inline-flex items-center rounded border px-1 font-mono text-[0.75em] leading-[1.6] transition-colors ${
    isActive
      ? 'border-accent bg-accent/15 text-accent'
      : 'border-accent/40 text-accent hover:border-accent hover:bg-accent/10'
  }`

  if (!interactive) {
    return <span className={className}>{n}</span>
  }

  return (
    <button
      type="button"
      className={`${className} focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent`}
      onMouseEnter={() => setActiveCitation(n)}
      onMouseLeave={() => setActiveCitation((current) => (current === n ? null : current))}
      onFocus={() => setActiveCitation(n)}
      onBlur={() => setActiveCitation((current) => (current === n ? null : current))}
      onClick={() => scrollToCard(n)}
      aria-describedby={`citation-card-${n}`}
    >
      {n}
    </button>
  )
}
