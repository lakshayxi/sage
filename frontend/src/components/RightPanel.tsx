import type { ChatResponse, CitationOut } from '../api/types'
import { CitationCard } from './CitationCard'
import { MetadataBar } from './MetadataBar'

function SectionLabel({ children }: { children: string }) {
  return (
    <h2 className="px-1 text-xs font-semibold tracking-wide text-text-muted uppercase">{children}</h2>
  )
}

interface RightPanelProps {
  result: ChatResponse | null
  citations?: CitationOut[]
}

export function RightPanel({ result, citations = result?.citations ?? [] }: RightPanelProps) {
  if (!result && citations.length === 0) {
    return (
      <div className="thin-scroll flex h-full flex-col gap-6 overflow-y-auto p-4">
        <p className="px-1 text-sm text-text-muted">
          Citations and query metadata will appear here once you ask a question.
        </p>
      </div>
    )
  }

  return (
    <div className="thin-scroll flex h-full flex-col gap-6 overflow-y-auto p-4">
      <section className="space-y-2">
        <SectionLabel>Citations</SectionLabel>
        {citations.length === 0 ? (
          <p className="px-1 text-sm text-text-muted">No citations for this answer.</p>
        ) : (
          <div className="space-y-2">
            {citations.map((citation) => (
              <CitationCard key={citation.n} citation={citation} />
            ))}
          </div>
        )}
      </section>

      {result ? (
        <section className="space-y-2">
          <SectionLabel>Query metadata</SectionLabel>
          <MetadataBar result={result} />
        </section>
      ) : (
        <p className="px-1 text-xs leading-relaxed text-text-muted">
          Saved citation snapshot from this conversation.
        </p>
      )}
    </div>
  )
}
