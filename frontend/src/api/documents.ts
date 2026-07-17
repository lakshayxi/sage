import type { DocumentOut } from './types'
import { demoKeyHeaders } from './session'

export async function getDocuments(): Promise<DocumentOut[]> {
  const response = await fetch('/documents', { headers: demoKeyHeaders() })
  if (!response.ok) throw new Error(`Failed to load documents (${response.status})`)
  return response.json()
}
