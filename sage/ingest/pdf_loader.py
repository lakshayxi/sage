"""PDF text extraction using PyMuPDF.

Reconstructs paragraph boundaries from block layout rather than trusting
literal blank lines in the extracted text: many PDF authoring tools emit one
text block per wrapped line rather than one block per paragraph. A vertical
gap between blocks that's meaningfully larger than the normal line-to-line
gap is treated as a paragraph break; smaller gaps are just wrapped lines
within the same paragraph.

Ported from the reference local-first project (`finresearch/ingest/pdf_loader.py`)
essentially unchanged — this specific heuristic (block-height-relative gap
ratio) was already proven there against real 10-K-shaped PDFs.
"""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

# gap / previous_block_height above this ratio is treated as a new paragraph.
PARAGRAPH_GAP_RATIO = 0.5


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str


def _reconstruct_paragraphs(blocks: list[tuple]) -> str:
    # block tuple: (x0, y0, x1, y1, text, block_no, block_type); type 0 = text.
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    text_blocks.sort(key=lambda b: (b[1], b[0]))

    paragraphs: list[str] = []
    current_lines: list[str] = []
    prev_y1: float | None = None
    prev_height: float | None = None

    for _x0, y0, _x1, y1, text, *_ in text_blocks:
        height = y1 - y0
        is_new_paragraph = False
        if prev_y1 is not None and prev_height:
            gap = y0 - prev_y1
            if gap / prev_height > PARAGRAPH_GAP_RATIO:
                is_new_paragraph = True

        if is_new_paragraph and current_lines:
            paragraphs.append(" ".join(current_lines))
            current_lines = []

        current_lines.extend(line.strip() for line in text.splitlines() if line.strip())
        prev_y1 = y1
        prev_height = height

    if current_lines:
        paragraphs.append(" ".join(current_lines))

    return "\n\n".join(paragraphs)


def load_pdf_pages(pdf_path: Path) -> list[PageText]:
    """Extract paragraph-reconstructed text from each page of a text-based PDF."""
    doc = fitz.open(pdf_path)
    try:
        pages = []
        for i, page in enumerate(doc):
            blocks = page.get_text("blocks")
            pages.append(PageText(page_number=i + 1, text=_reconstruct_paragraphs(blocks)))
        return pages
    finally:
        doc.close()
