import type { ChatResponse } from '../api/types'
import { CitationCard } from './CitationCard'
import { MetadataBar } from './MetadataBar'
import { RelatedQuestions } from './RelatedQuestions'

function SectionLabel({ children }: { children: string }) {
  return (
    <h2 className="px-1 text-xs font-semibold tracking-wide text-text-muted uppercase">{children}</h2>
  )
}

export function RightPanel({ result }: { result: ChatResponse | null }) {
  if (!result) {
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
        {result.citations.length === 0 ? (
          <p className="px-1 text-sm text-text-muted">No citations for this answer.</p>
        ) : (
          <div className="space-y-2">
            {result.citations.map((citation) => (
              <CitationCard key={citation.n} citation={citation} />
            ))}
          </div>
        )}
      </section>

      <section className="space-y-2">
        <SectionLabel>Related questions</SectionLabel>
        <RelatedQuestions />
      </section>

      <section className="space-y-2">
        <SectionLabel>Query metadata</SectionLabel>
        <MetadataBar result={result} />
      </section>
    </div>
  )
}
