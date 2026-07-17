import type { ConversationSummaryOut, DocumentOut } from '../api/types'
import { CompanyFilter } from './CompanyFilter'
import { FilingsList } from './FilingsList'
import { ConversationList } from './ConversationList'

interface SidebarProps {
  companies: string[]
  selectedCompanies: string[]
  onToggleCompany: (company: string) => void
  documents: DocumentOut[]
  conversations: ConversationSummaryOut[]
  activeConversationId: number | null
  onSelectConversation: (id: number) => void
  onNewConversation: () => void
}

function SectionLabel({ children }: { children: string }) {
  return (
    <h2 className="px-1 text-xs font-semibold tracking-wide text-text-muted uppercase">{children}</h2>
  )
}

export function Sidebar({
  companies,
  selectedCompanies,
  onToggleCompany,
  documents,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewConversation,
}: SidebarProps) {
  return (
    <div className="thin-scroll flex h-full flex-col gap-6 overflow-y-auto p-4">
      <div>
        <p className="font-display text-xl text-text">Sage</p>
        <p className="text-xs text-text-muted">Financial research workspace</p>
      </div>

      <button
        type="button"
        onClick={onNewConversation}
        className="flex items-center justify-center gap-1.5 rounded-lg border border-border bg-panel px-3 py-2 text-sm font-medium text-text transition-colors hover:border-accent/60 hover:text-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
      >
        + New conversation
      </button>

      <section className="space-y-2">
        <SectionLabel>Companies</SectionLabel>
        <CompanyFilter companies={companies} selected={selectedCompanies} onToggle={onToggleCompany} />
      </section>

      <section className="space-y-2">
        <SectionLabel>Filings</SectionLabel>
        <FilingsList documents={documents} />
      </section>

      <section className="min-h-0 flex-1 space-y-2">
        <SectionLabel>Conversations</SectionLabel>
        <ConversationList
          conversations={conversations}
          activeId={activeConversationId}
          onSelect={onSelectConversation}
        />
      </section>
    </div>
  )
}
