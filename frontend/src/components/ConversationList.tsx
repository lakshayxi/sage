import type { ConversationSummaryOut } from '../api/types'

interface ConversationListProps {
  conversations: ConversationSummaryOut[]
  activeId: number | null
  onSelect: (id: number) => void
}

export function ConversationList({ conversations, activeId, onSelect }: ConversationListProps) {
  if (conversations.length === 0) {
    return <p className="px-1 text-sm text-text-muted">No conversations yet.</p>
  }

  return (
    <ul className="space-y-0.5">
      {conversations.map((conversation) => {
        const isActive = conversation.id === activeId
        return (
          <li key={conversation.id}>
            <button
              type="button"
              onClick={() => onSelect(conversation.id)}
              className={`w-full truncate rounded px-1.5 py-1.5 text-left text-sm transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent ${
                isActive
                  ? 'border-l-2 border-accent bg-accent/10 text-text'
                  : 'border-l-2 border-transparent text-text-muted hover:bg-border/60 hover:text-text'
              }`}
            >
              {conversation.title || 'Untitled conversation'}
            </button>
          </li>
        )
      })}
    </ul>
  )
}
