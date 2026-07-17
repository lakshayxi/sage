// Mirrors api/schemas.py. Field names/shapes are kept identical to the
// backend response JSON on purpose -- no remapping layer.

export interface CitationOut {
  n: number
  chunk_id: number
  text: string
  page_number: number | null
  company: string | null
  fiscal_year: string | null
  doc_type: string | null
  filename: string
}

export interface LatencyOut {
  retrieval_ms: number
  generation_ms: number
  total_ms: number
}

export interface TokensOut {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export interface ChatResponse {
  schema_version: number
  answer: string
  citations: CitationOut[]
  model: string
  latency_ms: LatencyOut
  tokens: TokensOut
  cache_hit: boolean
  cost_usd: number
  session_id: number | null
}

// Citations as persisted on a historical message (sage/db/conversations.py
// via api/routes/chat.py's _persist_turn) -- a smaller field set than
// CitationOut, missing text/fiscal_year/doc_type.
export interface StoredCitation {
  n: number
  chunk_id: number
  filename: string
  page_number: number | null
  company: string | null
}

export interface ConversationSummaryOut {
  schema_version: number
  id: number
  title: string | null
  created_at: string
}

export interface MessageOut {
  id: number
  role: 'user' | 'assistant'
  content: string
  citations: StoredCitation[] | null
  created_at: string
}

export interface ConversationDetailResponse {
  schema_version: number
  id: number
  title: string | null
  created_at: string
  messages: MessageOut[]
}

export interface DocumentOut {
  schema_version: number
  id: number
  filename: string
  title: string | null
  company: string | null
  fiscal_year: string | null
  doc_type: string | null
  page_count: number
  status: string
  ingested_at: string
}
