"""Fixed token-window chunking, paragraph-boundary aware.

Token counts are approximated by word count (whitespace split) rather than a
model-specific tokenizer -- Gemini's exact tokenizer isn't exposed client-side
the way a local model's is, and an approximation is all this chunker needs to
produce consistently-sized, paragraph-respecting windows. Ported from the
reference project's `finresearch/ingest/chunker.py` with the same algorithm,
with one deliberate departure: chunks are never allowed to cross a page
boundary (see `chunk_pages`'s docstring) -- the reference project's chunker
predates Sage's per-chunk citation requirement, where a chunk's single
`page_number` must be an exact, trustworthy citation target, not just "the
page the chunk started on."
"""

from dataclasses import dataclass

from sage.ingest.pdf_loader import PageText


@dataclass
class Paragraph:
    page_number: int
    text: str
    char_start: int
    char_end: int


@dataclass
class Chunk:
    chunk_index: int
    text: str
    page_number: int
    token_count: int
    char_start: int
    char_end: int


def _word_count(text: str) -> int:
    return len(text.split())


def _paragraphs_from_page(page: PageText) -> list[Paragraph]:
    paragraphs = []
    cursor = 0
    for raw_para in page.text.split("\n\n"):
        para = raw_para.strip()
        if not para:
            continue
        start = cursor
        end = cursor + len(para)
        paragraphs.append(Paragraph(page.page_number, para, start, end))
        cursor = end + 2  # account for the "\n\n" separator
    return paragraphs


def _split_oversized_paragraph(
    para: Paragraph, chunk_tokens: int, overlap_tokens: int
) -> list[Paragraph]:
    """Split a paragraph whose word count alone exceeds `chunk_tokens` into
    bounded, overlapping word-count windows.

    Without this, a single oversized paragraph (a dense legal/risk-factor
    block with no blank-line breaks, common in real 10-Ks) would flow
    straight through the paragraph-boundary-respecting logic below as one
    unsplit unit -- `current` starts empty for it, so the `current_words +
    para_words > chunk_tokens` guard never fires, and it becomes one
    unbounded chunk. Each returned window is guaranteed <= chunk_tokens
    words, so `chunk_pages` can no longer produce an unbounded chunk this way.
    """
    words = para.text.split()
    if len(words) <= chunk_tokens:
        return [para]

    step = max(chunk_tokens - overlap_tokens, 1)
    windows = []
    start = 0
    while start < len(words):
        end = min(start + chunk_tokens, len(words))
        window_text = " ".join(words[start:end])
        windows.append(Paragraph(para.page_number, window_text, para.char_start, para.char_end))
        if end == len(words):
            break
        start += step
    return windows


def _chunk_page_paragraphs(
    paragraphs: list[Paragraph],
    chunk_tokens: int,
    overlap_tokens: int,
    start_index: int,
) -> list[Chunk]:
    """Chunk one page's paragraphs into ~chunk_tokens-word windows with
    overlap. `start_index` continues chunk_index numbering across pages."""
    chunks: list[Chunk] = []
    current: list[Paragraph] = []
    current_words = 0

    def flush():
        nonlocal current
        if not current:
            return
        text = "\n\n".join(p.text for p in current)
        chunks.append(
            Chunk(
                chunk_index=start_index + len(chunks),
                text=text,
                page_number=current[0].page_number,
                token_count=current_words,
                char_start=current[0].char_start,
                char_end=current[-1].char_end,
            )
        )

    for para in paragraphs:
        para_words = _word_count(para.text)

        if para_words > chunk_tokens:
            # Oversized paragraph: flush whatever's pending first, then emit
            # each pre-split window directly as its own chunk -- each window
            # is already <= chunk_tokens and windows already overlap each
            # other via _split_oversized_paragraph's step, so they don't go
            # through the paragraph-accumulation logic below at all (which
            # assumes paragraphs are meaningfully smaller than chunk_tokens
            # and would otherwise carry a whole ~chunk_tokens-sized window
            # forward as "overlap", blowing the next chunk's budget).
            flush()
            current = []
            current_words = 0
            for window in _split_oversized_paragraph(para, chunk_tokens, overlap_tokens):
                chunks.append(
                    Chunk(
                        chunk_index=start_index + len(chunks),
                        text=window.text,
                        page_number=window.page_number,
                        token_count=_word_count(window.text),
                        char_start=window.char_start,
                        char_end=window.char_end,
                    )
                )
            continue

        if current and current_words + para_words > chunk_tokens:
            flush()
            # Start the new chunk with an overlap tail from the chunk just closed.
            overlap: list[Paragraph] = []
            overlap_words = 0
            for p in reversed(current):
                w = _word_count(p.text)
                if overlap_words + w > overlap_tokens and overlap:
                    break
                overlap.insert(0, p)
                overlap_words += w
            current = overlap
            current_words = overlap_words

        current.append(para)
        current_words += para_words

    flush()
    return chunks


def chunk_pages(
    pages: list[PageText],
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Chunk extracted page text into ~chunk_tokens-word windows with overlap.

    Paragraph-boundary aware: paragraphs are never split mid-way unless a
    single paragraph alone exceeds chunk_tokens (see
    `_split_oversized_paragraph`). Chunks never cross a page boundary --
    each page is chunked independently (overlap included), so a chunk's
    single `page_number` is always the exact page all of its text came
    from, not just the page the chunk happened to start on. This makes
    per-chunk citations trustworthy: a citation naming page N is guaranteed
    to actually be on page N, never spilling onto page N+1.
    """
    chunks: list[Chunk] = []
    for page in pages:
        paragraphs = _paragraphs_from_page(page)
        chunks.extend(_chunk_page_paragraphs(paragraphs, chunk_tokens, overlap_tokens, len(chunks)))
    return chunks
