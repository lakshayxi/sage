import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { MainPanel } from './MainPanel'

describe('MainPanel empty state', () => {
  it('offers three filing-specific sample questions and explains comparison filters', () => {
    const html = renderToStaticMarkup(
      <MainPanel turns={[]} onAsk={vi.fn()} onStop={vi.fn()} isBusy={false} />,
    )

    expect(html.match(/type="button"/g)).toHaveLength(3)
    expect(html).toContain('Apple · performance')
    expect(html).toContain('Microsoft · investment')
    expect(html).toContain('Three-company comparison')
    expect(html).toContain('Select Apple, Microsoft, and NVIDIA in Companies before asking.')
    expect(html).toContain('Choose a prompt to place it in the question bar.')
  })

  it('shows a stop action while an answer is being generated', () => {
    const html = renderToStaticMarkup(
      <MainPanel turns={[]} onAsk={vi.fn()} onStop={vi.fn()} isBusy={true} />,
    )

    expect(html).toContain('Stop generating')
    expect(html).not.toContain('type="submit"')
  })
})
