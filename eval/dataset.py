"""Hand-curated Q&A eval set grounded in the three real, ingested SEC filings
under `data/raw/` (`Apple_FY25_filing.pdf`, `Microsoft_FY25_filing.pdf`,
`NVIDIA_FY26_filing.pdf` -- genuine 10-Ks pulled from EDGAR, not synthetic
fixtures like the reference project's).

Every `expected_amount_millions` / `expected_keywords` value below was
verified by reading the actual filing text (via PyMuPDF, not guessed or
recalled from memory) -- specifically the Consolidated Statements of
Income/Operations and segment-reporting tables:

- Apple FY2025 (10-K, p.29 in the PDF): Total net sales $416,161M, Net
  income $112,010M; net sales by category (p.23): iPhone $209,586M is the
  largest single category.
- Microsoft FY2025 (10-K, p.50): Total revenue $281,724M, Net income
  $101,832M, R&D expense $32,488M.
- NVIDIA FY2026 (fiscal year ended Jan 25, 2026; 10-K p.51): Revenue
  $215,938M, Net income $120,067M; segment revenue: Compute & Networking
  $193,479M vs. Graphics $22,459M; operating income $130,387M.

`companies` matches `generate_answer`'s actual plural, list-shaped parameter
(`sage/generation/answer_engine.py`) directly -- no translation layer to a
singular field like the reference project's `EvalItem.company`, since Sage's
retrieval signature was plural from the start (see
`sage/retrieval/retriever.py`'s module docstring).
"""

from dataclasses import dataclass, field

# Maps each ingested company to its source filename, used by
# `eval.scoring` to check that citations on an answerable item actually
# point back to the right document (not just *a* document).
COMPANY_FILENAMES: dict[str, str] = {
    "Apple": "Apple_FY25_filing.pdf",
    "Microsoft": "Microsoft_FY25_filing.pdf",
    "NVIDIA": "NVIDIA_FY26_filing.pdf",
}


@dataclass
class EvalItem:
    id: str
    question: str
    companies: list[str] | None = None
    fiscal_year: str | None = None
    doc_type: str | None = None
    answerable: bool = True
    # Human-readable note only, for the printed/CSV report -- not itself
    # parsed for scoring.
    expected_answer: str = ""
    # Numeric correctness check: the answer is scored correct if it contains
    # a dollar figure within `tolerance` (relative) of this value, in
    # millions. None if this item isn't a single-number lookup.
    expected_amount_millions: float | None = None
    tolerance: float = 0.02
    # Keyword correctness check: every string here must appear
    # (case-insensitive) in the answer. Used for qualitative/comparison
    # questions where "which one" matters more than an exact figure, and
    # combined with `expected_amount_millions` (AND) when both are set.
    expected_keywords: list[str] = field(default_factory=list)
    # Filenames citations are allowed to point at. None => derive from
    # `companies` (via COMPANY_FILENAMES), or "any of the three ingested
    # filings" if `companies` is also None (cross-company/unfiltered).
    allowed_filenames: list[str] | None = None


EVAL_ITEMS: list[EvalItem] = [
    # --- Apple FY25 ---
    EvalItem(
        id="apple-revenue-fy25",
        question="What was Apple's total net sales in fiscal year 2025?",
        companies=["Apple"],
        fiscal_year="FY25",
        expected_answer="$416,161 million (~$416.2 billion).",
        expected_amount_millions=416_161.0,
    ),
    EvalItem(
        id="apple-netincome-fy25",
        question="What was Apple's net income in fiscal year 2025?",
        companies=["Apple"],
        fiscal_year="FY25",
        expected_answer="$112,010 million.",
        expected_amount_millions=112_010.0,
    ),
    EvalItem(
        id="apple-segment-fy25",
        question="Which Apple product category had the highest net sales in fiscal year 2025?",
        companies=["Apple"],
        fiscal_year="FY25",
        expected_answer="iPhone, at $209,586 million.",
        expected_keywords=["iPhone"],
    ),
    # --- Microsoft FY25 ---
    EvalItem(
        id="msft-revenue-fy25",
        question="What was Microsoft's total revenue in fiscal year 2025?",
        companies=["Microsoft"],
        fiscal_year="FY25",
        expected_answer="$281,724 million.",
        expected_amount_millions=281_724.0,
    ),
    EvalItem(
        id="msft-netincome-fy25",
        question="What was Microsoft's net income in fiscal year 2025?",
        companies=["Microsoft"],
        fiscal_year="FY25",
        expected_answer="$101,832 million.",
        expected_amount_millions=101_832.0,
    ),
    EvalItem(
        id="msft-rnd-fy25",
        question="How much did Microsoft spend on research and development in fiscal year 2025?",
        companies=["Microsoft"],
        fiscal_year="FY25",
        expected_answer="$32,488 million.",
        expected_amount_millions=32_488.0,
    ),
    # --- NVIDIA FY26 ---
    EvalItem(
        id="nvda-revenue-fy26",
        question="What was NVIDIA's total revenue in fiscal year 2026?",
        companies=["NVIDIA"],
        fiscal_year="FY26",
        expected_answer="$215,938 million.",
        expected_amount_millions=215_938.0,
    ),
    EvalItem(
        id="nvda-netincome-fy26",
        question="What was NVIDIA's net income in fiscal year 2026?",
        companies=["NVIDIA"],
        fiscal_year="FY26",
        expected_answer="$120,067 million.",
        expected_amount_millions=120_067.0,
    ),
    EvalItem(
        id="nvda-segment-fy26",
        question=(
            "Which of NVIDIA's two reportable segments, Compute & Networking or Graphics, "
            "generated more revenue in fiscal year 2026?"
        ),
        companies=["NVIDIA"],
        fiscal_year="FY26",
        expected_answer="Compute & Networking, at $193,479 million vs. Graphics $22,459 million.",
        expected_amount_millions=193_479.0,
        expected_keywords=["Compute"],
    ),
    # --- Cross-company comparisons (comparison mode: companies has 2-3 entries,
    # exercising retrieve_hybrid's round-robin multi-company merge path -- see
    # sage/retrieval/retriever.py) ---
    EvalItem(
        id="cross-revenue-leader",
        question=(
            "Among Apple, Microsoft, and NVIDIA, which company reported the highest total "
            "revenue in its most recent fiscal year filing?"
        ),
        companies=["Apple", "Microsoft", "NVIDIA"],
        expected_answer=(
            "Apple, at $416,161M, ahead of Microsoft ($281,724M) and NVIDIA ($215,938M)."
        ),
        expected_keywords=["Apple"],
    ),
    EvalItem(
        id="cross-netincome-leader",
        question=(
            "Among Apple, Microsoft, and NVIDIA, which company reported the highest net income "
            "in its most recent fiscal year filing?"
        ),
        companies=["Apple", "Microsoft", "NVIDIA"],
        expected_answer=(
            "NVIDIA, at $120,067M, ahead of Apple ($112,010M) and Microsoft ($101,832M) -- note "
            "this is the opposite ranking from total revenue, a real check that the model isn't "
            "just pattern-matching 'biggest company = biggest number'."
        ),
        expected_keywords=["NVIDIA"],
    ),
    EvalItem(
        id="cross-operating-income",
        question=(
            "Between Microsoft and NVIDIA, which company reported higher operating income in "
            "its most recent fiscal year filing?"
        ),
        companies=["Microsoft", "NVIDIA"],
        expected_answer=(
            "NVIDIA, at $130,387M, versus Microsoft's $128,528M -- deliberately a close call "
            "(~1.4% apart) as a harder retrieval/generation precision stress test."
        ),
        expected_keywords=["NVIDIA"],
    ),
    # --- Deliberately out-of-corpus (grounding/refusal check: the corpus only
    # has Apple, Microsoft, and NVIDIA -- these ask about companies with no
    # ingested filing at all, verifying the relevance gate declines rather
    # than hallucinating a plausible-looking figure) ---
    EvalItem(
        id="unans-tesla-revenue",
        question="What was Tesla's total revenue in fiscal year 2025?",
        answerable=False,
        expected_answer="Not answerable -- no Tesla filing is ingested (only Apple/MSFT/NVIDIA).",
    ),
    EvalItem(
        id="unans-amazon-netincome",
        question="What was Amazon's net income in fiscal year 2025?",
        answerable=False,
        expected_answer="Not answerable -- no Amazon filing is ingested (only Apple/MSFT/NVIDIA).",
    ),
]
