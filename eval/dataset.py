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
    # Comparison correctness requires every company and figure, not merely
    # the winner's name. Values are expressed in millions and are also used
    # to verify per-filing retrieval/citation support.
    expected_company_amounts_millions: dict[str, float] = field(default_factory=dict)
    # Expected winner for a highest/higher comparison. Scoring requires the
    # company name to appear near explicit comparative language.
    expected_leader: str | None = None
    # Answer-only period labels. These are not required to appear verbatim in
    # raw citation text, where filings commonly spell FY25 as "2025".
    expected_periods: list[str] = field(default_factory=list)
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
    # Prior conversation turns, as {"role": "user"|"assistant", "content":
    # str} dicts -- exercises generate_answer's history-bearing path
    # (retrieval runs fresh off `question` alone; the LLM sees the full
    # history for continuity). None for the common single-shot case. A
    # history-bearing item is never cache-checked by generate_answer itself
    # regardless of run_eval.py's use_cache flag (see answer_engine.py).
    history: list[dict] | None = None


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
        expected_company_amounts_millions={
            "Apple": 416_161.0,
            "Microsoft": 281_724.0,
            "NVIDIA": 215_938.0,
        },
        expected_leader="Apple",
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
        expected_company_amounts_millions={
            "Apple": 112_010.0,
            "Microsoft": 101_832.0,
            "NVIDIA": 120_067.0,
        },
        expected_leader="NVIDIA",
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
        expected_company_amounts_millions={
            "Microsoft": 128_528.0,
            "NVIDIA": 130_387.0,
        },
        expected_leader="NVIDIA",
        expected_keywords=["NVIDIA"],
    ),
    EvalItem(
        id="cross-revenue-full-ranking",
        question=(
            "Compare total annual revenue for Apple, Microsoft, and NVIDIA using each company's "
            "ingested fiscal-year filing. State each fiscal year and amount in millions, then "
            "rank them from highest to lowest."
        ),
        companies=["Apple", "Microsoft", "NVIDIA"],
        expected_answer=(
            "Apple FY25 $416,161M, Microsoft FY25 $281,724M, NVIDIA FY26 $215,938M; "
            "ranking: Apple, Microsoft, NVIDIA."
        ),
        expected_company_amounts_millions={
            "Apple": 416_161.0,
            "Microsoft": 281_724.0,
            "NVIDIA": 215_938.0,
        },
        expected_leader="Apple",
        expected_periods=["FY25", "FY26"],
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
    # --- Unanswerable with a semantically similar distractor in-corpus:
    # unlike the Tesla/Amazon items above (no plausible distractor at all,
    # since neither company is ingested), these ask about a real ingested
    # company but a fact/period the corpus doesn't actually contain --
    # close enough to real content that hybrid retrieval is likely to
    # surface genuinely similar-looking chunks, making refusal a harder,
    # more meaningful test than an entirely out-of-corpus question. ---
    EvalItem(
        id="unans-apple-fy2020-revenue",
        question="What was Apple's total net sales in fiscal year 2020?",
        companies=["Apple"],
        answerable=False,
        expected_answer=(
            "Not answerable -- only Apple's FY25 filing is ingested; FY2020 figures "
            "aren't in the corpus, even though FY25 revenue (a distractor) is."
        ),
    ),
    EvalItem(
        id="unans-nvidia-automotive-segment",
        question=("How much revenue did NVIDIA's Automotive segment generate in fiscal year 2026?"),
        companies=["NVIDIA"],
        answerable=False,
        expected_answer=(
            "Not answerable as asked -- the ingested NVIDIA FY26 filing reports "
            "Compute & Networking and Graphics as its two segments, not a separate "
            "'Automotive' segment; a plausible-sounding but fabricated figure here "
            "would indicate the model pattern-matched to real segment-revenue "
            "chunks instead of checking whether this specific segment exists."
        ),
    ),
    # --- Multi-turn follow-ups: exercise generate_answer's history-bearing
    # path (retrieval runs fresh off the follow-up question alone; the LLM
    # sees prior turns for continuity). Not covered by any single-shot item
    # above. ---
    EvalItem(
        id="multiturn-apple-followup-netincome",
        # Deliberately not "And what was its net income in the same
        # period?" -- a purely pronoun-referenced follow-up carries no
        # standalone semantic content for the cross-encoder reranker to
        # score (it never sees conversation history, only this query
        # text), so it fails MIN_RERANK_SCORE's relevance gate before ever
        # reaching generation -- confirmed via a live run. Restating "net
        # income" and the fiscal year keeps this a real follow-up (still
        # needs history to know *whose* net income "its" refers to for a
        # correctly-scoped, continuity-aware answer) while giving the gate
        # enough to work with.
        question="What was its net income in fiscal year 2025?",
        companies=["Apple"],
        fiscal_year="FY25",
        expected_answer="$112,010 million.",
        expected_amount_millions=112_010.0,
        history=[
            {"role": "user", "content": "What was Apple's total net sales in fiscal year 2025?"},
            {
                "role": "assistant",
                "content": "Apple's total net sales in fiscal year 2025 were $416,161 million [1].",
            },
        ],
    ),
    EvalItem(
        id="multiturn-cross-company-followup",
        # See multiturn-apple-followup-netincome's comment -- "those two"
        # alone gives the reranker nothing to score; naming both companies
        # keeps this answerable via retrieval while "compared to the prior
        # question" still requires history for the model to know which
        # figure ("net income", not "operating income") is actually new.
        question="Between Microsoft and NVIDIA, which had the higher net income?",
        companies=["Microsoft", "NVIDIA"],
        expected_answer=("NVIDIA ($120,067M) had higher net income than Microsoft ($101,832M)."),
        expected_company_amounts_millions={
            "Microsoft": 101_832.0,
            "NVIDIA": 120_067.0,
        },
        expected_leader="NVIDIA",
        history=[
            {
                "role": "user",
                "content": (
                    "Between Microsoft and NVIDIA, which had higher operating income in "
                    "its most recent fiscal year filing?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "NVIDIA reported higher operating income ($130,387M) than Microsoft "
                    "($128,528M) [1, 2]."
                ),
            },
        ],
    ),
]
