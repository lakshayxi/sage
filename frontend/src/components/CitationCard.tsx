import { useState } from 'react'
import type { CitationOut } from '../api/types'
import { useCitationHighlight } from '../context/CitationHighlightContext'

export function CitationCard({ citation }: { citation: CitationOut }) {
  const { activeCitation, setActiveCitation, registerCard } = useCitationHighlight()
  const [expanded, setExpanded] = useState(false)
  const isActive = activeCitation === citation.n

  return (
    <div
      id={`citation-card-${citation.n}`}
      ref={(el) => registerCard(citation.n, el)}
      onMouseEnter={() => setActiveCitation(citation.n)}
      onMouseLeave={() => setActiveCitation((current) => (current === citation.n ? null : current))}
      className={`rounded-lg border bg-panel px-3 py-2.5 transition-colors ${
        isActive ? 'border-l-4 border-accent bg-accent/10' : 'border-border border-l-4'
      }`}
    >
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        onFocus={() => setActiveCitation(citation.n)}
        onBlur={() => setActiveCitation((current) => (current === citation.n ? null : current))}
        className="flex w-full items-start gap-2 text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
      >
        <span className="mt-0.5 inline-flex shrink-0 items-center rounded border border-accent/40 px-1 font-mono text-xs text-accent">
          {citation.n}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm text-text">{citation.filename}</span>
          <span className="mt-0.5 flex flex-wrap gap-x-2 font-mono text-xs text-text-muted">
            {citation.company && <span>{citation.company}</span>}
            {citation.fiscal_year && <span>FY{citation.fiscal_year}</span>}
            {citation.page_number != null && <span>p.{citation.page_number}</span>}
          </span>
        </span>
        <span className="mt-0.5 shrink-0 font-mono text-xs text-text-muted">{expanded ? '▾' : '▸'}</span>
      </button>
      {expanded && (
        <p className="mt-2 border-t border-border pt-2 text-sm leading-relaxed text-text-muted">
          {citation.text}
        </p>
      )}
    </div>
  )
}
