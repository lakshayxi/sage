"""Fixed token-window chunking, paragraph-boundary aware.

Token counts are approximated by word count (whitespace split) rather than a
model-specific tokenizer -- Gemini's exact tokenizer isn't exposed client-side
the way a local model's is, and an approximation is all this chunker needs to
produce consistently-sized, paragraph-respecting windows. Ported from the
reference project's `finresearch/ingest/chunker.py` with the same algorithm.
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


def _paragraphs_from_pages(pages: list[PageText]) -> list[Paragraph]:
    paragraphs = []
    cursor = 0
    for page in pages:
        for raw_para in page.text.split("\n\n"):
            para = raw_para.strip()
            if not para:
                continue
            start = cursor
            end = cursor + len(para)
            paragraphs.append(Paragraph(page.page_number, para, start, end))
            cursor = end + 2  # account for the "\n\n" separator
        cursor += 2  # separator between pages
    return paragraphs


def chunk_pages(
    pages: list[PageText],
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Chunk extracted page text into ~chunk_tokens-word windows with overlap.

    Paragraph-boundary aware: paragraphs are never split mid-way unless a
    single paragraph alone exceeds chunk_tokens.
    """
    paragraphs = _paragraphs_from_pages(pages)
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
                chunk_index=len(chunks),
                text=text,
                page_number=current[0].page_number,
                token_count=current_words,
                char_start=current[0].char_start,
                char_end=current[-1].char_end,
            )
        )

    for para in paragraphs:
        para_words = _word_count(para.text)

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
