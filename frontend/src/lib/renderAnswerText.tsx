import { Children, type ReactNode } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { CitationMarker } from '../components/CitationMarker'

interface AnswerMarkdownProps {
  text: string
  interactive: boolean
}

const CITATION_GROUP_RE = /\[(\s*\d+(?:\s*,\s*\d+)*)\]/g

function renderCitationText(text: string, interactive: boolean): ReactNode {
  const nodes: ReactNode[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null
  let groupIndex = 0
  CITATION_GROUP_RE.lastIndex = 0

  while ((match = CITATION_GROUP_RE.exec(text))) {
    if (match.index > lastIndex) nodes.push(text.slice(lastIndex, match.index))

    const citationNumbers = match[1].split(',').map((value) => Number(value.trim()))
    nodes.push(
      <span
        key={`citation-group-${match.index}-${groupIndex}`}
        className="inline-flex items-baseline gap-1 whitespace-nowrap"
      >
        {citationNumbers.map((n, citationIndex) => (
          <span key={`${n}-${citationIndex}`} className="inline-flex items-baseline gap-1">
            {citationIndex > 0 && <span className="text-text-muted">,</span>}
            <CitationMarker n={n} interactive={interactive} />
          </span>
        ))}
      </span>,
    )

    lastIndex = CITATION_GROUP_RE.lastIndex
    groupIndex += 1
  }

  if (lastIndex === 0) return text
  if (lastIndex < text.length) nodes.push(text.slice(lastIndex))
  return nodes
}

function renderCitations(children: ReactNode, interactive: boolean): ReactNode {
  return Children.map(children, (child) =>
    typeof child === 'string' ? renderCitationText(child, interactive) : child,
  )
}

function markdownComponents(interactive: boolean): Components {
  const inline = (children: ReactNode) => renderCitations(children, interactive)

  return {
    h1: ({ children }) => (
      <h1 className="font-display text-2xl font-semibold tracking-tight">{inline(children)}</h1>
    ),
    h2: ({ children }) => (
      <h2 className="font-display text-xl font-semibold tracking-tight">{inline(children)}</h2>
    ),
    h3: ({ children }) => (
      <h3 className="font-display text-lg font-semibold tracking-tight">{inline(children)}</h3>
    ),
    h4: ({ children }) => <h4 className="font-semibold text-text">{inline(children)}</h4>,
    h5: ({ children }) => <h5 className="font-semibold text-text">{inline(children)}</h5>,
    h6: ({ children }) => <h6 className="font-semibold text-text">{inline(children)}</h6>,
    p: ({ children }) => <p className="leading-relaxed">{inline(children)}</p>,
    strong: ({ children }) => (
      <strong className="font-semibold text-text">{inline(children)}</strong>
    ),
    em: ({ children }) => <em>{inline(children)}</em>,
    del: ({ children }) => <del>{inline(children)}</del>,
    a: ({ children, href }) => (
      <a
        href={href}
        className="text-accent underline decoration-accent/50 underline-offset-2 hover:text-accent-hover"
      >
        {inline(children)}
      </a>
    ),
    ul: ({ children }) => <ul className="list-disc space-y-1 pl-5">{children}</ul>,
    ol: ({ children }) => <ol className="list-decimal space-y-1 pl-5">{children}</ol>,
    li: ({ children }) => <li className="leading-relaxed">{inline(children)}</li>,
    blockquote: ({ children }) => (
      <blockquote className="border-l-2 border-accent/50 pl-3 text-text-muted">
        {children}
      </blockquote>
    ),
    table: ({ children }) => (
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full border-collapse text-sm">{children}</table>
      </div>
    ),
    thead: ({ children }) => <thead className="bg-panel">{children}</thead>,
    th: ({ children }) => (
      <th className="border-b border-border px-3 py-2 text-left font-semibold text-text">
        {inline(children)}
      </th>
    ),
    td: ({ children }) => (
      <td className="border-t border-border px-3 py-2 align-top leading-relaxed">
        {inline(children)}
      </td>
    ),
  }
}

/** Safely renders streamed assistant Markdown with interactive inline citations. */
export function AnswerMarkdown({ text, interactive }: AnswerMarkdownProps) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={markdownComponents(interactive)}
      skipHtml
    >
      {text}
    </ReactMarkdown>
  )
}
