import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { CitationHighlightProvider } from '../context/CitationHighlightContext'
import { AnswerMarkdown } from './renderAnswerText'

function renderAnswer(text: string, interactive = true): string {
  return renderToStaticMarkup(
    <CitationHighlightProvider>
      <AnswerMarkdown text={text} interactive={interactive} />
    </CitationHighlightProvider>,
  )
}

describe('AnswerMarkdown', () => {
  it('renders headings, bold text, lists, and GFM tables', () => {
    const markup = renderAnswer(`### Apple

**Revenue** increased.

- First item
- Second item

1. One
2. Two

| Company | Revenue |
| --- | ---: |
| Apple | $1 |

### Comparison`)

    expect(markup).toContain('<h3')
    expect(markup).toContain('>Apple</h3>')
    expect(markup).toContain('>Comparison</h3>')
    expect(markup).toContain('<strong')
    expect(markup).toContain('<ul')
    expect(markup).toContain('<ol')
    expect(markup).toContain('<table')
    expect(markup).toContain('<th')
    expect(markup).toContain('<td')
    expect(markup).not.toContain('### Apple')
  })

  it('turns single and grouped citations into individual interactive markers', () => {
    const markup = renderAnswer('**Margins** improved [6, 7, 9], while revenue grew [3].')

    expect(markup.match(/<button/g)).toHaveLength(4)
    expect(markup).toContain('aria-describedby="citation-card-6"')
    expect(markup).toContain('aria-describedby="citation-card-7"')
    expect(markup).toContain('aria-describedby="citation-card-9"')
    expect(markup).toContain('aria-describedby="citation-card-3"')
    expect(markup).not.toContain('[6, 7, 9]')
  })

  it('keeps historical-answer citation markers non-interactive', () => {
    const markup = renderAnswer('Evidence [11, 12].', false)

    expect(markup).not.toContain('<button')
    expect(markup).toContain('>11</span>')
    expect(markup).toContain('>12</span>')
  })

  it('does not render raw HTML from assistant output', () => {
    const markup = renderAnswer('Before <script>alert("x")</script> after <span>unsafe</span>.')

    expect(markup).not.toContain('<script')
    expect(markup).not.toContain('<span>unsafe</span>')
  })

  it('renders incomplete streamed Markdown without requiring a completed answer', () => {
    const markup = renderAnswer('### NVIDIA\n\nRevenue is still arriving')

    expect(markup).toContain('>NVIDIA</h3>')
    expect(markup).toContain('Revenue is still arriving')
  })
})
