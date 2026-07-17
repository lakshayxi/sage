import type { DocumentOut } from '../api/types'

export function FilingsList({ documents }: { documents: DocumentOut[] }) {
  if (documents.length === 0) {
    return <p className="px-1 text-sm text-text-muted">No filings ingested yet.</p>
  }

  return (
    <ul className="divide-y divide-border">
      {documents.map((doc) => (
        <li key={doc.id} className="py-2 px-1">
          <p className="truncate text-sm text-text" title={doc.filename}>
            {doc.filename}
          </p>
          <p className="mt-0.5 flex flex-wrap items-center gap-x-2 font-mono text-xs text-text-muted">
            <span>{doc.company ?? 'Unknown'}</span>
            {doc.fiscal_year && <span>FY{doc.fiscal_year}</span>}
            <span>{doc.page_count}p</span>
            {doc.status !== 'ready' && <span className="text-negative">{doc.status}</span>}
          </p>
        </li>
      ))}
    </ul>
  )
}
