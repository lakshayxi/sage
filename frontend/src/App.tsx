import { useCallback, useEffect, useMemo, useState } from 'react'
import { getDocuments } from './api/documents'
import { listConversations } from './api/conversations'
import type { ConversationSummaryOut, DocumentOut } from './api/types'
import { useTheme } from './hooks/useTheme'
import { useChatSession } from './hooks/useChatSession'
import { CitationHighlightProvider } from './context/CitationHighlightContext'
import { Sidebar } from './components/Sidebar'
import { MainPanel } from './components/MainPanel'
import { RightPanel } from './components/RightPanel'
import { ThemeToggle } from './components/ThemeToggle'

function App() {
  const { theme, toggleTheme } = useTheme()
  const [documents, setDocuments] = useState<DocumentOut[]>([])
  const [conversations, setConversations] = useState<ConversationSummaryOut[]>([])
  const [selectedCompanies, setSelectedCompanies] = useState<string[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [citationsOpen, setCitationsOpen] = useState(false)

  const refreshConversations = useCallback(() => {
    listConversations()
      .then(setConversations)
      .catch(() => setLoadError('Could not load conversation history.'))
  }, [])

  useEffect(() => {
    getDocuments()
      .then(setDocuments)
      .catch(() => setLoadError('Could not load filings from the backend.'))
    refreshConversations()
  }, [refreshConversations])

  const { conversationId, turns, isBusy, ask, startNewConversation, resumeConversation } =
    useChatSession(refreshConversations)

  const companies = useMemo(
    () => Array.from(new Set(documents.map((d) => d.company).filter((c): c is string => Boolean(c)))).sort(),
    [documents],
  )

  const latestResult = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i -= 1) {
      const turn = turns[i]
      if (!turn.fromHistory && turn.result) return turn.result
    }
    return null
  }, [turns])

  function toggleCompany(company: string) {
    setSelectedCompanies((current) =>
      current.includes(company) ? current.filter((c) => c !== company) : [...current, company],
    )
  }

  function handleAsk(query: string) {
    setSidebarOpen(false)
    ask(query, { companies: selectedCompanies }).catch(() =>
      setLoadError('Could not reach the server to ask that question.'),
    )
  }

  function handleSelectConversation(id: number) {
    setSidebarOpen(false)
    resumeConversation(id).catch(() => setLoadError('Could not load that conversation.'))
  }

  function handleNewConversation() {
    setSidebarOpen(false)
    startNewConversation()
  }

  return (
    <CitationHighlightProvider>
      <div className="flex h-svh flex-col bg-bg text-text">
        <header className="flex shrink-0 items-center justify-between border-b border-border px-4 py-3 lg:justify-end lg:px-6">
          <button
            type="button"
            onClick={() => setSidebarOpen(true)}
            className="rounded-lg border border-border px-2.5 py-1.5 text-sm text-text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent lg:hidden"
            aria-label="Open companies and conversations"
          >
            ☰
          </button>
          <p className="font-display text-lg text-text lg:hidden">Sage</p>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setCitationsOpen(true)}
              className="rounded-lg border border-border px-2.5 py-1.5 font-mono text-xs text-text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent lg:hidden"
              aria-label="Open citations"
            >
              [n]
            </button>
            <ThemeToggle theme={theme} onToggle={toggleTheme} />
          </div>
        </header>

        {loadError && (
          <p className="shrink-0 border-b border-border bg-negative/10 px-4 py-2 text-center text-sm text-negative">
            {loadError}
          </p>
        )}

        <div className="flex min-h-0 flex-1">
          <aside
            className={`fixed inset-y-0 left-0 z-20 w-72 border-r border-border bg-panel transition-transform lg:static lg:z-auto lg:translate-x-0 ${
              sidebarOpen ? 'translate-x-0' : '-translate-x-full'
            }`}
          >
            <Sidebar
              companies={companies}
              selectedCompanies={selectedCompanies}
              onToggleCompany={toggleCompany}
              documents={documents}
              conversations={conversations}
              activeConversationId={conversationId}
              onSelectConversation={handleSelectConversation}
              onNewConversation={handleNewConversation}
            />
          </aside>
          {sidebarOpen && (
            <button
              type="button"
              aria-label="Close menu"
              onClick={() => setSidebarOpen(false)}
              className="fixed inset-0 z-10 bg-black/60 lg:hidden"
            />
          )}

          <main className="min-h-0 min-w-0 flex-1">
            <MainPanel turns={turns} onAsk={handleAsk} isBusy={isBusy} />
          </main>

          <aside
            className={`fixed inset-y-0 right-0 z-20 w-80 border-l border-border bg-panel transition-transform lg:static lg:z-auto lg:translate-x-0 ${
              citationsOpen ? 'translate-x-0' : 'translate-x-full'
            }`}
          >
            <RightPanel result={latestResult} />
          </aside>
          {citationsOpen && (
            <button
              type="button"
              aria-label="Close citations"
              onClick={() => setCitationsOpen(false)}
              className="fixed inset-0 z-10 bg-black/60 lg:hidden"
            />
          )}
        </div>
      </div>
    </CitationHighlightProvider>
  )
}

export default App
