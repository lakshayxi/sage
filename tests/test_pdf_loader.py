from sage.ingest.pdf_loader import _reconstruct_paragraphs


def _block(x0, y0, x1, y1, text, block_no=0, block_type=0):
    return (x0, y0, x1, y1, text, block_no, block_type)


def test_small_gap_between_blocks_stays_one_paragraph():
    # Two lines of the same paragraph: gap (2) is small relative to block height (10).
    blocks = [
        _block(0, 0, 100, 10, "First line of a paragraph.\n"),
        _block(0, 12, 100, 22, "Second line, still the same paragraph.\n"),
    ]
    result = _reconstruct_paragraphs(blocks)
    assert "\n\n" not in result
    assert "First line" in result and "Second line" in result


def test_large_gap_between_blocks_starts_new_paragraph():
    # Gap (40) is much larger than the previous block's height (10) -> new paragraph.
    blocks = [
        _block(0, 0, 100, 10, "First paragraph.\n"),
        _block(0, 50, 100, 60, "Second paragraph.\n"),
    ]
    result = _reconstruct_paragraphs(blocks)
    assert result == "First paragraph.\n\nSecond paragraph."


def test_non_text_blocks_are_ignored():
    blocks = [
        _block(0, 0, 100, 10, "Text block.\n", block_type=0),
        _block(0, 20, 100, 120, "", block_type=1),  # e.g. an image block
    ]
    result = _reconstruct_paragraphs(blocks)
    assert result == "Text block."


def test_blank_text_blocks_are_skipped():
    blocks = [
        _block(0, 0, 100, 10, "   \n", block_type=0),
        _block(0, 20, 100, 30, "Real content.\n", block_type=0),
    ]
    result = _reconstruct_paragraphs(blocks)
    assert result == "Real content."
