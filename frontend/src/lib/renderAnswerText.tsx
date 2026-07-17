import type { ReactNode } from 'react'
import { CitationMarker } from '../components/CitationMarker'

// The model's actual output shape (see sage/generation/prompts.py) is
// plain paragraphs, occasional **bold** company headings in comparison
// answers, occasional bullet/numbered lists, and inline [n] citations --
// this covers exactly that, not general-purpose markdown.

const CITATION_RE = /\[(\d+)\]/g
const BOLD_RE = /\*\*(.+?)\*\*/g

function renderBold(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null
  let i = 0
  BOLD_RE.lastIndex = 0
  while ((match = BOLD_RE.exec(text))) {
    if (match.index > lastIndex) nodes.push(text.slice(lastIndex, match.index))
    nodes.push(
      <strong key={`${keyPrefix}-b${i}`} className="font-semibold text-text">
        {match[1]}
      </strong>,
    )
    lastIndex = BOLD_RE.lastIndex
    i += 1
  }
  if (lastIndex < text.length) nodes.push(text.slice(lastIndex))
  return nodes
}

function renderInline(text: string, interactive: boolean, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null
  let i = 0
  CITATION_RE.lastIndex = 0
  while ((match = CITATION_RE.exec(text))) {
    if (match.index > lastIndex) {
      nodes.push(...renderBold(text.slice(lastIndex, match.index), `${keyPrefix}-t${i}`))
    }
    nodes.push(
      <CitationMarker key={`${keyPrefix}-c${i}`} n={Number(match[1])} interactive={interactive} />,
    )
    lastIndex = CITATION_RE.lastIndex
    i += 1
  }
  if (lastIndex < text.length) {
    nodes.push(...renderBold(text.slice(lastIndex), `${keyPrefix}-t${i}`))
  }
  return nodes
}

/** Renders LLM answer text as paragraphs/lists/bold with inline citation markers. */
export function renderAnswerText(text: string, interactive: boolean): ReactNode[] {
  const blocks = text.trim().split(/\n{2,}/)
  return blocks.map((block, blockIndex) => {
    const lines = block.split('\n').filter((line) => line.trim().length > 0)
    const isBulletList = lines.length > 0 && lines.every((line) => /^[-*]\s+/.test(line.trim()))
    const isNumberedList = lines.length > 0 && lines.every((line) => /^\d+\.\s+/.test(line.trim()))

    if (isBulletList) {
      return (
        <ul key={blockIndex} className="list-disc space-y-1 pl-5">
          {lines.map((line, i) => (
            <li key={i}>
              {renderInline(line.trim().replace(/^[-*]\s+/, ''), interactive, `${blockIndex}-${i}`)}
            </li>
          ))}
        </ul>
      )
    }

    if (isNumberedList) {
      return (
        <ol key={blockIndex} className="list-decimal space-y-1 pl-5">
          {lines.map((line, i) => (
            <li key={i}>
              {renderInline(line.trim().replace(/^\d+\.\s+/, ''), interactive, `${blockIndex}-${i}`)}
            </li>
          ))}
        </ol>
      )
    }

    return (
      <p key={blockIndex} className="leading-relaxed">
        {renderInline(block, interactive, `${blockIndex}`)}
      </p>
    )
  })
}
