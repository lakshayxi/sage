import { describe, expect, it } from 'vitest'
import { getFilingSourceUrl } from './filingSources'

describe('getFilingSourceUrl', () => {
  it('maps each curated filing to its authoritative SEC archive page', () => {
    expect(getFilingSourceUrl('Apple_FY25_filing.pdf')).toContain('sec.gov/Archives/edgar')
    expect(getFilingSourceUrl('Microsoft_FY25_filing.pdf')).toContain('sec.gov/Archives/edgar')
    expect(getFilingSourceUrl('NVIDIA_FY26_filing.pdf')).toContain('sec.gov/Archives/edgar')
  })

  it('does not invent a link for an unknown upload', () => {
    expect(getFilingSourceUrl('unknown.pdf')).toBeNull()
  })
})
