interface CompanyFilterProps {
  companies: string[]
  selected: string[]
  onToggle: (company: string) => void
}

// Multi-select from the start (companies is already list-shaped in
// ChatRequest) even though v1 only wires single-company queries end-to-end --
// see CLAUDE brief. Selecting zero companies means "search everything."
export function CompanyFilter({ companies, selected, onToggle }: CompanyFilterProps) {
  if (companies.length === 0) {
    return <p className="px-1 text-sm text-text-muted">No companies ingested yet.</p>
  }

  return (
    <ul className="space-y-0.5">
      {companies.map((company) => {
        const isSelected = selected.includes(company)
        return (
          <li key={company}>
            <label className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-sm hover:bg-border/60">
              <input
                type="checkbox"
                checked={isSelected}
                onChange={() => onToggle(company)}
                className="h-3.5 w-3.5 accent-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
              />
              <span className={isSelected ? 'text-text' : 'text-text-muted'}>{company}</span>
            </label>
          </li>
        )
      })}
    </ul>
  )
}
