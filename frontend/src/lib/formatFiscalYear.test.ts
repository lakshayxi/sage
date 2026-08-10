import { describe, expect, it } from 'vitest'
import { formatFiscalYear } from './formatFiscalYear'

describe('formatFiscalYear', () => {
  it.each([
    ['25', 'FY25'],
    ['2025', 'FY2025'],
    ['FY25', 'FY25'],
    ['fy2026', 'FY2026'],
    ['FYFY25', 'FY25'],
    ['FY FY26', 'FY26'],
    ['FY-FY-2027', 'FY2027'],
  ])('formats %s with exactly one FY prefix', (input, expected) => {
    expect(formatFiscalYear(input)).toBe(expected)
  })

  it.each([null, undefined, '', '   '])('returns null for missing year %s', (input) => {
    expect(formatFiscalYear(input)).toBeNull()
  })
})
