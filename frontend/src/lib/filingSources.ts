const SEC_FILING_URLS: Record<string, string> = {
  'Apple_FY25_filing.pdf':
    'https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm',
  'Microsoft_FY25_filing.pdf':
    'https://www.sec.gov/Archives/edgar/data/789019/000095017025100235/msft-20250630.htm',
  'NVIDIA_FY26_filing.pdf':
    'https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm',
}

export function getFilingSourceUrl(filename: string): string | null {
  return SEC_FILING_URLS[filename] ?? null
}
