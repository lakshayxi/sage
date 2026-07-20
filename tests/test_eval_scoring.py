"""Unit tests for eval/scoring.py -- pure functions, no network, no live
Gemini call (unlike eval/run_eval.py itself, which is inherently a
live-network tool and isn't exercised by this file or by `pytest tests/`)."""

from eval.dataset import EvalItem
from eval.scoring import (
    allowed_filenames_for,
    citation_text_supports_expected,
    extract_amounts_millions,
    keywords_match,
    leader_match,
    numeric_match,
    refusal_detected,
    score_item,
)


def test_extract_amounts_millions_handles_dollar_and_million_unit():
    assert extract_amounts_millions("Apple's net sales were $416,161 million.") == [416_161.0]


def test_extract_amounts_millions_handles_billion_phrasing():
    assert extract_amounts_millions("roughly $416.2 billion in net sales") == [416_200.0]


def test_extract_amounts_millions_handles_raw_dollar_figure():
    assert extract_amounts_millions("$112,010,000,000 in net income") == [112_010.0]


def test_extract_amounts_millions_ignores_bare_numbers():
    # Citation markers and fiscal years shouldn't be mistaken for dollar figures.
    assert extract_amounts_millions("See [1, 5] for fiscal year 2025 details.") == []


def test_extract_amounts_millions_finds_multiple_figures():
    text = "Revenue was $215,938 million and net income was $120,067 million."
    assert extract_amounts_millions(text) == [215_938.0, 120_067.0]


def test_numeric_match_within_tolerance():
    assert numeric_match("Net sales were $416,161 million.", 416_161.0, tolerance=0.02)


def test_numeric_match_rejects_wrong_figure():
    assert not numeric_match("Net sales were $300,000 million.", 416_161.0, tolerance=0.02)


def test_numeric_match_accepts_rounded_billion_phrasing_within_tolerance():
    # 416.2B == 416,200M, within 2% of 416,161M.
    assert numeric_match("about $416.2 billion", 416_161.0, tolerance=0.02)


def test_keywords_match_is_case_insensitive_and_requires_all():
    assert keywords_match("Compute & Networking led with $193,479M.", ["Compute"])
    assert not keywords_match("Graphics revenue was higher.", ["Compute"])


def test_leader_match_requires_comparative_language_near_company():
    assert leader_match("Apple reported the highest revenue of the three.", "Apple")
    assert leader_match("The highest revenue was reported by Apple.", "Apple")
    assert leader_match("Ranking from highest to lowest: 1. Apple", "Apple")
    assert not leader_match("Apple and Microsoft were compared.", "Apple")


def test_refusal_detected_matches_exact_gate_message():
    assert refusal_detected(
        "I don't have information in the ingested documents that's relevant to this question."
    )


def test_refusal_detected_matches_rephrased_refusal():
    assert refusal_detected("The provided context does not mention Tesla's revenue.")


def test_refusal_detected_false_for_confident_answer():
    assert not refusal_detected("Apple's net sales were $416,161 million.")


def test_allowed_filenames_for_single_company():
    item = EvalItem(id="x", question="q", companies=["Apple"])
    assert allowed_filenames_for(item) == {"Apple_FY25_filing.pdf"}


def test_allowed_filenames_for_unfiltered_item_is_all_three():
    item = EvalItem(id="x", question="q", companies=None)
    assert allowed_filenames_for(item) == {
        "Apple_FY25_filing.pdf",
        "Microsoft_FY25_filing.pdf",
        "NVIDIA_FY26_filing.pdf",
    }


def test_score_item_passes_on_correct_grounded_answer():
    item = EvalItem(
        id="apple-revenue",
        question="q",
        companies=["Apple"],
        expected_amount_millions=416_161.0,
    )
    result = score_item(item, "Apple's net sales were $416,161 million.", ["Apple_FY25_filing.pdf"])
    assert result.correct
    assert result.grounded
    assert result.passed


def test_score_item_fails_when_answer_wrong_even_if_grounded():
    item = EvalItem(
        id="apple-revenue",
        question="q",
        companies=["Apple"],
        expected_amount_millions=416_161.0,
    )
    result = score_item(item, "Apple's net sales were $1 million.", ["Apple_FY25_filing.pdf"])
    assert not result.correct
    assert not result.passed


def test_score_item_fails_when_citation_points_at_wrong_company():
    item = EvalItem(
        id="apple-revenue",
        question="q",
        companies=["Apple"],
        expected_amount_millions=416_161.0,
    )
    result = score_item(
        item, "Apple's net sales were $416,161 million.", ["Microsoft_FY25_filing.pdf"]
    )
    assert result.correct
    assert not result.grounded
    assert not result.passed


def test_score_item_fails_when_no_citations_at_all():
    item = EvalItem(
        id="apple-revenue",
        question="q",
        companies=["Apple"],
        expected_amount_millions=416_161.0,
    )
    result = score_item(item, "Apple's net sales were $416,161 million.", [])
    assert not result.grounded


def test_comparison_scoring_requires_all_company_values_and_source_files():
    item = EvalItem(
        id="cross-revenue",
        question="q",
        companies=["Apple", "Microsoft"],
        expected_company_amounts_millions={"Apple": 416_161.0, "Microsoft": 281_724.0},
        expected_leader="Apple",
    )
    answer = (
        "Apple reported the highest revenue at $416,161 million; "
        "Microsoft reported $281,724 million."
    )
    result = score_item(
        item,
        answer,
        ["Apple_FY25_filing.pdf"],
        citation_texts=["Total net sales $ 416,161"],
    )

    assert result.correct
    assert not result.grounded
    assert not result.text_supported
    assert not result.passed


def test_comparison_scoring_passes_with_per_company_values_and_support():
    item = EvalItem(
        id="cross-revenue",
        question="q",
        companies=["Apple", "Microsoft"],
        expected_company_amounts_millions={"Apple": 416_161.0, "Microsoft": 281_724.0},
        expected_leader="Apple",
    )
    answer = (
        "Apple reported the highest revenue at $416,161 million; "
        "Microsoft reported $281,724 million."
    )
    result = score_item(
        item,
        answer,
        ["Apple_FY25_filing.pdf", "Microsoft_FY25_filing.pdf"],
        citation_texts=["Total net sales $ 416,161", "Total revenue $ 281,724"],
    )

    assert result.passed


def test_score_item_unanswerable_passes_on_refusal_with_no_citations():
    item = EvalItem(id="unans", question="q", answerable=False)
    result = score_item(item, "The provided context does not mention Tesla's revenue.", [])
    assert result.passed


def test_score_item_unanswerable_fails_if_it_fabricates_an_answer():
    item = EvalItem(id="unans", question="q", answerable=False)
    result = score_item(item, "Tesla's revenue was $100,000 million.", ["Apple_FY25_filing.pdf"])
    assert not result.correct
    assert not result.grounded
    assert not result.passed


def test_citation_text_supports_expected_true_when_figure_present():
    item = EvalItem(id="apple-revenue", question="q", expected_amount_millions=416_161.0)
    assert citation_text_supports_expected(
        item, ["Total net sales were $416,161 million for fiscal 2025."]
    )


def test_citation_text_supports_expected_false_when_figure_absent():
    """Regression test: a citation pointing at an allowed-filename chunk
    that doesn't actually contain the expected figure (e.g. the model cited
    an irrelevant page of the right filing) must not be treated as
    supporting the answer just because the filename check passed."""
    item = EvalItem(id="apple-revenue", question="q", expected_amount_millions=416_161.0)
    assert not citation_text_supports_expected(
        item, ["Apple Inc. is headquartered in Cupertino, California."]
    )


def test_citation_text_supports_expected_true_for_raw_financial_table_text():
    """Regression test found via a live run against the real corpus: a
    cited chunk that's raw 10-K financial-statement-table text (scale
    stated once in a caption, e.g. "(In millions...)", not spelled out
    next to every number the way generated prose does) used to be scored
    as NOT supporting an expected figure it actually states, because
    numeric_match's raw-dollar-figure heuristic (built for prose) divided
    the bare "$ 416,161" by another 1,000,000. This is the exact chunk text
    from a real live citation against the ingested Apple 10-K."""
    item = EvalItem(id="apple-revenue", question="q", expected_amount_millions=416_161.0)
    assert citation_text_supports_expected(
        item,
        [
            "Total net sales $ 416,161 $ 391,035 $ 383,285\n\n"
            "Portion of total net sales that was included in deferred "
            "revenue as of the beginning of the period $ 8,229 $ 7,728 $ 8,169"
        ],
    )


def test_citation_text_supports_expected_false_with_no_citation_texts():
    item = EvalItem(id="apple-revenue", question="q", expected_amount_millions=416_161.0)
    assert not citation_text_supports_expected(item, [])


def test_citation_text_supports_expected_checks_keywords_too():
    item = EvalItem(id="nvda-segment", question="q", expected_keywords=["Compute"])
    assert citation_text_supports_expected(item, ["Compute & Networking revenue grew."])
    assert not citation_text_supports_expected(item, ["Graphics revenue grew."])


def test_citation_text_supports_expected_true_for_unanswerable_item():
    item = EvalItem(id="unans", question="q", answerable=False)
    assert citation_text_supports_expected(item, [])


def test_score_item_text_supported_defaults_true_when_not_checked():
    """Callers that don't pass citation_texts (e.g. existing tests above,
    or any caller predating this check) must see unchanged behavior."""
    item = EvalItem(id="apple-revenue", question="q", expected_amount_millions=416_161.0)
    result = score_item(item, "Apple's net sales were $416,161 million.", ["Apple_FY25_filing.pdf"])
    assert result.text_supported is True
    assert result.passed


def test_score_item_fails_when_citation_text_does_not_support_answer():
    """Regression test: filename-only grounding used to pass even when the
    cited chunk's actual text has nothing to do with the expected figure."""
    item = EvalItem(id="apple-revenue", question="q", expected_amount_millions=416_161.0)
    result = score_item(
        item,
        "Apple's net sales were $416,161 million.",
        ["Apple_FY25_filing.pdf"],
        citation_texts=["Apple Inc. designs, manufactures, and markets smartphones."],
    )
    assert result.correct
    assert result.grounded
    assert not result.text_supported
    assert not result.passed


def test_score_item_unanswerable_flags_hallucinated_grounding_even_with_refusal_text():
    # A refusal that nonetheless cites something is inconsistent -- still
    # worth surfacing as not cleanly grounded, even though `correct` passes.
    item = EvalItem(id="unans", question="q", answerable=False)
    result = score_item(
        item,
        "The provided context does not mention Tesla's revenue.",
        ["Apple_FY25_filing.pdf"],
    )
    assert result.correct
    assert not result.grounded
    assert not result.passed


def test_comparison_scoring_rejects_company_amount_swaps():
    item = EvalItem(
        id="comparison",
        question="q",
        companies=["Apple", "Microsoft"],
        expected_company_amounts_millions={
            "Apple": 416_161.0,
            "Microsoft": 281_724.0,
        },
    )
    result = score_item(
        item,
        "Apple reported $281,724 million. Microsoft reported $416,161 million.",
        ["Apple_FY25_filing.pdf", "Microsoft_FY25_filing.pdf"],
    )

    assert not result.correct
    assert not result.passed


def test_comparison_scoring_associates_amount_below_markdown_company_heading():
    item = EvalItem(
        id="comparison",
        question="q",
        companies=["Apple", "Microsoft"],
        expected_company_amounts_millions={
            "Apple": 416_161.0,
            "Microsoft": 281_724.0,
        },
    )
    result = score_item(
        item,
        "### Apple\nFor FY2025, revenue was $416,161 million.\n\n"
        "### Microsoft\nFor FY2025, revenue was $281,724 million.",
        ["Apple_FY25_filing.pdf", "Microsoft_FY25_filing.pdf"],
    )

    assert result.correct
