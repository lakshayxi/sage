import type { DocumentOut } from '../api/types'
import { formatFiscalYear } from '../lib/formatFiscalYear'
import { getFilingSourceUrl } from '../lib/filingSources'

export function FilingsList({ documents }: { documents: DocumentOut[] }) {
  if (documents.length === 0) {
    return <p className="px-1 text-sm text-text-muted">No filings ingested yet.</p>
  }

  return (
    <ul className="divide-y divide-border">
      {documents.map((doc) => {
        const fiscalYear = formatFiscalYear(doc.fiscal_year)
        const sourceUrl = getFilingSourceUrl(doc.filename)
        return (
          <li key={doc.id} className="py-2 px-1">
            <p className="truncate text-sm text-text" title={doc.filename}>
              {doc.filename}
            </p>
            <p className="mt-0.5 flex flex-wrap items-center gap-x-2 font-mono text-xs text-text-muted">
              <span>{doc.company ?? 'Unknown'}</span>
              {fiscalYear && <span>{fiscalYear}</span>}
              <span>{doc.page_count}p</span>
              {doc.status !== 'ready' && <span className="text-negative">{doc.status}</span>}
            </p>
            {sourceUrl && (
              <a
                href={sourceUrl}
                target="_blank"
                rel="noreferrer"
                className="mt-1 inline-block font-mono text-[0.68rem] text-accent underline-offset-4 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                SEC source ↗
              </a>
            )}
          </li>
        )
      })}
    </ul>
  )
}
