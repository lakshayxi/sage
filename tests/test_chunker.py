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


def test_chunks_never_cross_page_boundaries():
    """Regression test: two short pages used to merge into a single chunk
    whose stored page_number was only the *starting* page, silently
    mis-attributing page 2's text to a page-1 citation. Each page must
    chunk independently, even when small enough to have fit together."""
    pages = [
        PageText(page_number=1, text=_make_paragraph("pageone", 10)),
        PageText(page_number=2, text=_make_paragraph("pagetwo", 10)),
    ]
    chunks = chunk_pages(pages, chunk_tokens=650, overlap_tokens=120)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert "pageone" in chunks[0].text
    assert "pagetwo" not in chunks[0].text
    assert chunks[1].page_number == 2
    assert "pagetwo" in chunks[1].text
    assert "pageone" not in chunks[1].text
    # chunk_index keeps counting across the page boundary, not reset per page.
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_exact_cited_page_is_correct_for_every_chunk():
    """Every chunk's page_number must exactly match the page all of its text
    actually came from -- the core provenance guarantee citations rely on."""
    pages = [
        PageText(page_number=1, text=_make_paragraph("alpha", 300)),
        PageText(page_number=2, text=_make_paragraph("beta", 300)),
        PageText(page_number=3, text=_make_paragraph("gamma", 300)),
    ]
    chunks = chunk_pages(pages, chunk_tokens=650, overlap_tokens=120)

    assert {c.page_number for c in chunks} == {1, 2, 3}
    for c in chunks:
        expected_word = {1: "alpha", 2: "beta", 3: "gamma"}[c.page_number]
        assert expected_word in c.text
        for other_word in {"alpha", "beta", "gamma"} - {expected_word}:
            assert other_word not in c.text


def test_oversized_single_paragraph_is_split_into_bounded_windows():
    """A single paragraph bigger than chunk_tokens (no blank-line breaks,
    e.g. a dense risk-factor block) used to flow through as one unsplit,
    unbounded chunk. It must instead split into bounded, overlapping windows."""
    huge_paragraph = _make_paragraph("word", 2000)
    pages = [PageText(page_number=1, text=huge_paragraph)]

    chunks = chunk_pages(pages, chunk_tokens=650, overlap_tokens=120)

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 650
        assert c.page_number == 1


def test_oversized_paragraph_windows_overlap_without_growing_unbounded():
    huge_paragraph = _make_paragraph("term", 1500)
    pages = [PageText(page_number=1, text=huge_paragraph)]

    chunks = chunk_pages(pages, chunk_tokens=650, overlap_tokens=120)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count <= 650
    # Consecutive windows share an overlapping tail, same guarantee as the
    # paragraph-level overlap test above.
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-1] in second_words[: len(first_words)]


def test_empty_pages_produce_no_chunks():
    assert chunk_pages([], chunk_tokens=650, overlap_tokens=120) == []
    assert (
        chunk_pages([PageText(page_number=1, text="")], chunk_tokens=650, overlap_tokens=120) == []
    )
