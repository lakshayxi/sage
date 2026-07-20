"""System prompt and context/citation templates for answer generation.

Two system prompts: a single-company `SYSTEM_PROMPT` (ported from the
reference project, same citation-fence contract) and a
`COMPARISON_SYSTEM_PROMPT` used for explicit caller-selected multi-company
scope. The comparison variant is a real product requirement, not internal
plumbing -- "Compare Apple vs Microsoft's capex" reads badly as a single
blended paragraph with interleaved, ambiguous citations, so it explicitly
instructs the model to structure the answer per-company before comparing.

NOTE on live validation (2026-07-17, real GEMINI_API_KEY, model=gemini-flash-latest):
both SYSTEM_PROMPT and COMPARISON_SYSTEM_PROMPT were exercised live and both
produced the exact ```citations fence format on the first try, with correct
per-chunk citation numbers and, for the comparison prompt, clean per-company
sections followed by an explicit comparison section -- see
`sage/generation/answer_engine.py`'s citation-parsing fallbacks (carried
over from a project built against llama3.1, which sometimes dropped the
fence or emitted the array unfenced) for the failure class those defend
against; that class was not observed in this session's live calls, so the
fallback paths themselves remain unverified against genuine Gemini output
(only the well-formed happy path was exercised).
"""

from sage.retrieval.retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer ONLY using the numbered "
    "context chunks below. If the context is insufficient, say so explicitly. "
    "Match the requested company, fiscal period, and accounting category exactly: "
    "never substitute a nearby period or reinterpret a market platform, product "
    "category, or other business label as a reportable segment. If that exact "
    "scope is unsupported, refuse the requested figure and do not cite unrelated "
    "or merely similar context; return an empty citations array. "
    "Cite every factual claim with [n] referencing the chunk number. Context "
    "chunks are prefixed with Company / Fiscal Year / Doc Type / Page."
)

COMPARISON_SYSTEM_PROMPT = (
    "You are a financial research assistant helping compare multiple "
    "companies. Answer ONLY using the numbered context chunks below, which "
    "are drawn from more than one company's documents -- each chunk is "
    "prefixed with its Company / Fiscal Year / Doc Type / Page. Structure "
    "your answer with one clearly labeled section per company (e.g. a "
    "heading or bolded company name) covering what the context says about "
    "that company, then end with a short explicit comparison across "
    "companies. Do not blend sources from different companies into a single "
    "unattributed sentence -- every factual claim must be citeable to "
    "exactly one company's chunk via [n]. If the context is insufficient for "
    "one or more companies, say so explicitly for each affected company "
    "rather than omitting it silently. Match each requested fiscal period and "
    "accounting category exactly: never substitute a nearby period or reinterpret "
    "a market platform, product category, or other business label as a reportable "
    "segment. For an unsupported scope, refuse that requested figure and do not "
    "cite unrelated or merely similar context."
)

CITATION_FORMAT_INSTRUCTION = (
    "After your answer, you MUST end your response with a fenced code block "
    "that starts with the exact line ```citations and ends with the exact "
    "line ```. Do not omit the triple backticks. Inside the fence, put a JSON "
    "array of the bracket numbers [n] you cited above -- nothing else. Do "
    "NOT include a chunk_id or any other field; the number alone is enough, "
    "since [n] always refers to the chunk labeled [n] above. Example of the "
    "exact required format:\n"
    "```citations\n"
    "[1, 3]\n"
    "```"
)


def is_comparison(chunks: list[RetrievedChunk]) -> bool:
    """True when the retrieved context spans more than one distinct company."""
    companies = {c.company for c in chunks if c.company}
    return len(companies) > 1


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = (
            f"[{i}] chunk_id={c.chunk_id} | Company: {c.company or 'unknown'} | "
            f"Fiscal Year: {c.fiscal_year or 'unknown'} | Doc Type: {c.doc_type or 'unknown'} | "
            f"Page: {c.page_number or 'unknown'}"
        )
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


def build_user_message(query: str, chunks: list[RetrievedChunk]) -> str:
    context = build_context_block(chunks)
    return f"Context chunks:\n\n{context}\n\nQuestion: {query}\n\n{CITATION_FORMAT_INSTRUCTION}"


def build_messages(
    query: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
    comparison_mode: bool | None = None,
) -> list[dict]:
    """Build the full role/content message list for a generation call.

    `comparison_mode` lets the answer engine select the prompt from explicit
    caller scope rather than incidental candidate metadata. None preserves
    the auto-detection behavior for direct callers. `history` (prior turns
    from an ongoing conversation, if any) is inserted between the system
    prompt and the new user message so a multi-turn session keeps
    conversational continuity. Retrieval itself is still always run fresh
    off just the latest query -- only the LLM sees the full history.
    """
    if comparison_mode is None:
        comparison_mode = is_comparison(chunks)
    system_prompt = COMPARISON_SYSTEM_PROMPT if comparison_mode else SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": build_user_message(query, chunks)})
    return messages
