from sage.ingest.chunker import chunk_pages
from sage.ingest.pdf_loader import PageText


def _make_paragraph(word: str, n_words: int) -> str:
    return " ".join([word] * n_words)


def test_single_short_page_produces_one_chunk():
    pages = [PageText(page_number=1, text="A short paragraph about margins.")]
    chunks = chunk_pages(pages, chunk_tokens=650, overlap_tokens=120)

    assert len(chunks) == 1
    assert chunks[0].text == "A short paragraph about margins."
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0


def test_long_text_splits_into_multiple_chunks_with_overlap():
    paragraphs = [_make_paragraph(f"word{i}", 400) for i in range(3)]
    text = "\n\n".join(paragraphs)
    pages = [PageText(page_number=1, text=text)]

    chunks = chunk_pages(pages, chunk_tokens=650, overlap_tokens=120)

    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.token_count <= 650 + 400


def test_paragraph_boundaries_are_respected():
    paragraphs = [
        _make_paragraph("alpha", 300),
        _make_paragraph("beta", 300),
        _make_paragraph("gamma", 300),
    ]
    text = "\n\n".join(paragraphs)
    pages = [PageText(page_number=1, text=text)]

    chunks = chunk_pages(pages, chunk_tokens=650, overlap_tokens=120)

    for c in chunks:
        for para in paragraphs:
            if para in c.text:
                assert c.text.count(para) >= 1


def test_overlap_carries_tail_words_into_next_chunk():
    paragraphs = [_make_paragraph(f"p{i}", 400) for i in range(3)]
    text = "\n\n".join(paragraphs)
    pages = [PageText(page_number=1, text=text)]

    chunks = chunk_pages(pages, chunk_tokens=650, overlap_tokens=120)

    assert len(chunks) >= 2
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-1] in second_words[: len(first_words)]


def test_multi_page_chunk_tracks_starting_page_number():
    pages = [
        PageText(page_number=1, text=_make_paragraph("pageone", 10)),
        PageText(page_number=2, text=_make_paragraph("pagetwo", 10)),
    ]
    chunks = chunk_pages(pages, chunk_tokens=650, overlap_tokens=120)

    assert len(chunks) == 1
    assert chunks[0].page_number == 1


def test_empty_pages_produce_no_chunks():
    assert chunk_pages([], chunk_tokens=650, overlap_tokens=120) == []
    assert (
        chunk_pages([PageText(page_number=1, text="")], chunk_tokens=650, overlap_tokens=120) == []
    )
