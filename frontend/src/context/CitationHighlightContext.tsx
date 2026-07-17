import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from 'react'

interface CitationHighlightValue {
  activeCitation: number | null
  setActiveCitation: Dispatch<SetStateAction<number | null>>
  // Lets a CitationCard register its own DOM node under its citation number,
  // so CitationMarker's click-to-scroll can look it up through the same
  // shared mechanism as hover/focus highlighting instead of an unenforced
  // `document.getElementById` + hardcoded id-string convention.
  registerCard: (n: number, el: HTMLElement | null) => void
  scrollToCard: (n: number) => void
}

const CitationHighlightContext = createContext<CitationHighlightValue | null>(null)

// Shared "which citation number is currently hovered/focused" state, so an
// inline [n] marker in the answer and its matching card in the right panel
// can highlight each other regardless of where they sit in the tree -- this
// is the product's signature interaction (see CLAUDE brief).
export function CitationHighlightProvider({ children }: { children: ReactNode }) {
  const [activeCitation, setActiveCitation] = useState<number | null>(null)
  const cardRefs = useRef(new Map<number, HTMLElement>())

  const registerCard = useCallback((n: number, el: HTMLElement | null) => {
    if (el) cardRefs.current.set(n, el)
    else cardRefs.current.delete(n)
  }, [])

  const scrollToCard = useCallback((n: number) => {
    cardRefs.current.get(n)?.scrollIntoView({ block: 'nearest' })
  }, [])

  const value = useMemo(
    () => ({ activeCitation, setActiveCitation, registerCard, scrollToCard }),
    [activeCitation, registerCard, scrollToCard],
  )
  return (
    <CitationHighlightContext.Provider value={value}>{children}</CitationHighlightContext.Provider>
  )
}

export function useCitationHighlight(): CitationHighlightValue {
  const context = useContext(CitationHighlightContext)
  if (!context) {
    throw new Error('useCitationHighlight must be used within a CitationHighlightProvider')
  }
  return context
}
