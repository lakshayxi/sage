/** Keep filing metadata readable when the API already includes an FY prefix. */
export function formatFiscalYear(value: string | null | undefined): string | null {
  if (!value) return null

  const trimmed = value.trim()
  if (!trimmed) return null

  // Historical ingests contain both `25` and `FY25` (and can therefore
  // arrive after earlier UI formatting as `FYFY25`). Strip every leading
  // FY marker before adding the single display prefix.
  const year = trimmed.replace(/^(?:FY[\s._:/-]*)+/i, '').trim()
  return year ? `FY${year}` : 'FY'
}
