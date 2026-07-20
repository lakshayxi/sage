import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { ChatResponse } from '../api/types'
import { CitationHighlightProvider } from '../context/CitationHighlightContext'
import { RightPanel } from './RightPanel'

const result: ChatResponse = {
  schema_version: 1,
  answer: 'Answer.',
  citations: [],
  model: 'test-model',
  latency_ms: {
    retrieval_ms: 1,
    generation_ms: 2,
    total_ms: 3,
  },
  tokens: {
    prompt_tokens: 4,
    completion_tokens: 5,
    total_tokens: 9,
  },
  cache_hit: false,
  cost_usd: 0,
  session_id: 1,
}

describe('RightPanel', () => {
  it('shows citations and metadata without unfinished related-question copy', () => {
    const markup = renderToStaticMarkup(
      <CitationHighlightProvider>
        <RightPanel result={result} />
      </CitationHighlightProvider>,
    )

    expect(markup).toContain('Citations')
    expect(markup).toContain('Query metadata')
    expect(markup).not.toContain('Related questions')
    expect(markup).not.toContain('coming in a future pass')
  })
})
