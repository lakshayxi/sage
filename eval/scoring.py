"""Pure, network-free scoring logic for `eval/run_eval.py`.

Deviation from the reference project: the reference used DeepEval with a
local Ollama judge model (three LLM-as-judge metrics: faithfulness, answer
relevancy, context precision). That doesn't transfer cleanly here:

1. Sage's generation is Gemini-only (see `sage/generation/gemini_client.py`),
   so a "local, free, unmetered" judge isn't available the way Ollama was --
   an LLM judge here would mean *doubling* the live Gemini calls this harness
   makes (one to generate the answer, one to judge it), against a free-tier
   key already documented (`docs/llm-engineer-work-log.md`, Issue 2) to have
   hit a hard, non-recoverable `429 RESOURCE_EXHAUSTED` wall once already.
2. This dataset is deliberately narrow (financial figures with one
   objectively correct value, verified by reading the source PDFs -- see
   `eval/dataset.py`), not open-ended summarization/reasoning output where a
   judge model earns its keep over a string/number check. A deterministic
   heuristic is *more* reliable here, not less: it can't misjudge a correct
   $112,010M as wrong the way an LLM judge occasionally hallucinates a
   miscomparison.
3. It avoids adding DeepEval's dependency footprint (~24 packages) for a
   harness explicitly scoped as basic/correctness-only (see task framing) --
   consistent with this project's existing "thin wrapper, no heavy framework
   unless the task needs it" style.

Each item is scored on two independent axes:
- `correct`: does the answer contain the right figure/keyword (or, for
  deliberately unanswerable items, a refusal rather than a fabricated one)?
- `grounded`: do the citations actually point at the filing(s) the question
  is about (or, for unanswerable items, are there no citations at all --
  i.e. nothing was passed off as support for a made-up answer)?

`passed = correct and grounded` in both cases.
"""

import re
from dataclasses import dataclass

from eval.dataset import COMPANY_FILENAMES, EvalItem

# Mirrors sage.generation.answer_engine.NO_RELEVANT_CONTEXT_ANSWER without
# importing that module here -- keeps this file's only dependency being
# `eval.dataset` (a plain dataclass module), so `tests/test_eval_scoring.py`
# stays fast and import-safe with zero risk of an unrelated import in the
# generation package (google-genai client construction, etc.) breaking a
# no-network unit test. The exact-match check below is intentionally
# supplemented by the looser regex so a *rephrased* refusal (the LLM
# declining in its own words, rather than hitting the pre-generation
# relevance gate) still scores as correct.
_EXACT_REFUSAL_TEXT = (
    "I don't have information in the ingested documents that's relevant to this question."
)

_REFUSAL_PATTERNS = [
    r"\bdon'?t have (?:enough |sufficient )?information\b",
    r"\bdoes not (?:contain|include|mention|state|provide)\b",
    r"\bdoesn'?t (?:contain|include|mention|state|provide)\b",
    r"\bno information\b",
    r"\bnot (?:mentioned|stated|provided|available|found|contained)\b",
    r"\bcannot find\b",
    r"\bunable to find\b",
    r"\bnot in the (?:ingested|provided|available)\b",
    r"\bnot part of the (?:ingested|provided|available)\b",
    r"\bcontext does not\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

# Requires a leading "$" or a trailing scale word to count as a monetary
# figure -- otherwise this would false-positive on citation markers ("[1,
# 5]"), page numbers, or the fiscal year itself ("2025").
_NUMBER = r"[\d]{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+"
_AMOUNT_RE = re.compile(rf"(\$\s*)?({_NUMBER})\s*(billion|bn|million|mn)?", re.IGNORECASE)
_CITATION_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_amounts_millions(text: str) -> list[float]:
    """Return every dollar figure found in `text`, normalized to millions.

    Requires either a leading "$" or a trailing "million"/"billion" (or
    "mn"/"bn") to treat a number as a monetary figure at all. When a "$"
    figure has no scale word (e.g. a raw "$416,161,000,000"), it's assumed
    to already be expressed in whole dollars and divided down to millions;
    when a scale word is present, that's used directly regardless of "$".
    """
    amounts = []
    for dollar, num_str, unit in _AMOUNT_RE.findall(text):
        if not dollar and not unit:
            continue  # bare number, e.g. a year or a citation index -- skip
        value = float(num_str.replace(",", ""))
        unit = unit.lower()
        if unit in ("billion", "bn"):
            millions = value * 1000
        elif unit in ("million", "mn"):
            millions = value
        else:
            millions = value / 1_000_000
        amounts.append(millions)
    return amounts


def numeric_match(text: str, expected_millions: float, tolerance: float) -> bool:
    if expected_millions == 0:
        return False
    return any(
        abs(amount - expected_millions) / abs(expected_millions) <= tolerance
        for amount in extract_amounts_millions(text)
    )


def keywords_match(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return all(keyword.lower() in lowered for keyword in keywords)


def leader_match(text: str, leader: str) -> bool:
    """Whether `leader` is tied to explicit comparative language."""
    escaped = re.escape(leader)
    comparative = r"(?:highest|higher|largest|leader|ranking|rank(?:ed|s)?\s+(?:first|#?1))"
    return bool(
        re.search(rf"\b{escaped}\b.{{0,120}}\b{comparative}\b", text, re.I | re.S)
        or re.search(rf"\b{comparative}\b.{{0,120}}\b{escaped}\b", text, re.I | re.S)
    )


def refusal_detected(text: str) -> bool:
    return text.strip() == _EXACT_REFUSAL_TEXT or bool(_REFUSAL_RE.search(text))


def allowed_filenames_for(item: EvalItem) -> set[str]:
    if item.allowed_filenames is not None:
        return set(item.allowed_filenames)
    if item.companies:
        return {COMPANY_FILENAMES[c] for c in item.companies if c in COMPANY_FILENAMES}
    return set(COMPANY_FILENAMES.values())


def source_text_contains_amount(text: str, expected_millions: float) -> bool:
    """Whether raw *source* text (a cited chunk, not the model's generated
    prose) states the expected figure directly in millions.

    `numeric_match`/`extract_amounts_millions` were built for and validated
    against the model's own answer text, which reliably writes a scale word
    next to every figure ("$416,161 million", "$416.2 billion"). Real 10-K
    financial statement tables don't -- they state their scale once in a
    caption ("CONSOLIDATED STATEMENTS OF OPERATIONS (In millions...)") and
    then list bare numbers like "$ 416,161" against every line. Naively
    reusing `numeric_match` against that raw table text mis-parses those
    bare figures as whole dollars (dividing by another 1,000,000) and
    reports them as *not* matching an expected value they actually do
    state -- confirmed against a real live citation from the ingested
    Apple 10-K, whose actual chunk text is "Total net sales $ 416,161 ...".
    This checks for the plain formatted number instead, sidestepping the
    unit-word-detection problem entirely for this specific (source-table,
    not prose) input.
    """
    if expected_millions == 0:
        return False
    if expected_millions == int(expected_millions):
        n = int(expected_millions)
        candidates = [f"{n:,}", str(n)]
    else:
        candidates = [f"{expected_millions:,.2f}", str(expected_millions)]
    return any(candidate in text for candidate in candidates)


def citation_text_supports_expected(
    item: EvalItem,
    citation_texts: list[str],
    citation_filenames: list[str] | None = None,
) -> bool:
    """Whether the *content* of the cited chunks actually contains/supports
    the expected value -- stronger than `grounded`'s filename-only check,
    which would still pass if the model cited a real, allowed-filename
    chunk that happens to say nothing about the actual expected figure
    (e.g. cites the right 10-K's cover page instead of its income
    statement). Not applicable to unanswerable items (nothing to support).

    Tries both the prose-oriented `numeric_match` (in case a cited chunk
    happens to be narrative MD&A text that does spell out "million") and
    `source_text_contains_amount` (the common case: a financial
    statement table stating its scale only in a caption) -- either is
    accepted as support.
    """
    if not item.answerable:
        return True
    if not citation_texts:
        return False
    combined = " ".join(citation_texts)
    checks = []
    if item.expected_company_amounts_millions:
        if citation_filenames is None:
            checks.extend(
                source_text_contains_amount(combined, expected)
                for expected in item.expected_company_amounts_millions.values()
            )
        else:
            for company, expected in item.expected_company_amounts_millions.items():
                filename = COMPANY_FILENAMES.get(company)
                company_text = " ".join(
                    text
                    for cited_filename, text in zip(citation_filenames, citation_texts, strict=True)
                    if cited_filename == filename
                )
                checks.append(
                    bool(company_text) and source_text_contains_amount(company_text, expected)
                )
    if item.expected_amount_millions is not None:
        checks.append(
            numeric_match(combined, item.expected_amount_millions, item.tolerance)
            or source_text_contains_amount(combined, item.expected_amount_millions)
        )
    if item.expected_keywords:
        checks.append(keywords_match(combined, item.expected_keywords))
    if not checks:
        return True
    return all(checks)


def citation_company_associations_valid(
    item: EvalItem,
    answer_text: str,
    citation_filenames: list[str],
    citation_texts: list[str],
    citation_numbers: list[int],
) -> bool:
    """Tie each comparison claim's inline markers to that company's evidence.

    Aggregate checks can prove that an answer contains the right amounts and
    that its citation set contains the right source chunks while still
    missing a damaging swap: Apple's amount marked with Microsoft's citation
    and vice versa. This validates the association within each bounded company
    section, using the resolved citation number rather than list position.
    """
    if not item.expected_company_amounts_millions:
        return True
    if not (len(citation_filenames) == len(citation_texts) == len(citation_numbers)):
        return False

    source_by_number = {
        number: (filename, text)
        for number, filename, text in zip(
            citation_numbers, citation_filenames, citation_texts, strict=True
        )
    }
    company_names = list(item.expected_company_amounts_millions)
    lowered_answer = answer_text.lower()
    for company, expected in item.expected_company_amounts_millions.items():
        company_start = lowered_answer.find(company.lower())
        if company_start < 0:
            return False
        company_end = min(len(answer_text), company_start + 600)
        for other_company in company_names:
            if other_company == company:
                continue
            position = lowered_answer.find(other_company.lower(), company_start + len(company))
            if position >= 0:
                company_end = min(company_end, position)

        marker_numbers = {
            int(number)
            for marker in _CITATION_MARKER_RE.finditer(answer_text[company_start:company_end])
            for number in marker.group(1).split(",")
        }
        expected_filename = COMPANY_FILENAMES.get(company)
        if not any(
            number in source_by_number
            and source_by_number[number][0] == expected_filename
            and source_text_contains_amount(source_by_number[number][1], expected)
            for number in marker_numbers
        ):
            return False
    return True


@dataclass
class ScoreResult:
    correct: bool
    grounded: bool
    text_supported: bool
    detail: str
    citation_association_valid: bool = True

    @property
    def passed(self) -> bool:
        return (
            self.correct
            and self.grounded
            and self.text_supported
            and self.citation_association_valid
        )


def score_item(
    item: EvalItem,
    answer_text: str,
    citation_filenames: list[str],
    citation_texts: list[str] | None = None,
    citation_numbers: list[int] | None = None,
) -> ScoreResult:
    """`citation_texts`, if given, is the text of each resolved citation
    (`Citation.text`) in the same order as `citation_filenames` -- used for
    the text_supported check above. Omitted (`None`, the default) skips
    that check entirely (`text_supported=True` unconditionally) rather than
    failing every caller that predates it, e.g. `tests/test_eval_scoring.py`
    exercising `score_item` with just filenames."""
    if not item.answerable:
        correct = refusal_detected(answer_text)
        grounded = len(citation_filenames) == 0
        detail = (
            f"refusal_detected={correct}, citations={citation_filenames or 'none'} "
            f"(expected none for an out-of-corpus question)"
        )
        return ScoreResult(correct=correct, grounded=grounded, text_supported=True, detail=detail)

    checks = []
    if item.expected_amount_millions is not None:
        checks.append(numeric_match(answer_text, item.expected_amount_millions, item.tolerance))
    if item.expected_keywords:
        checks.append(keywords_match(answer_text, item.expected_keywords))
    if item.expected_company_amounts_millions:
        company_names = list(item.expected_company_amounts_millions)
        for company, expected in item.expected_company_amounts_millions.items():
            # Associate each figure with its company rather than accepting the
            # right bag of names and numbers in the wrong pairings. Bound the
            # company section at the next named company; markdown headings put
            # the amount on the following line, so a newline is not a boundary.
            company_start = answer_text.lower().find(company.lower())
            if company_start < 0:
                checks.append(False)
                continue
            company_end = min(len(answer_text), company_start + 600)
            for other_company in company_names:
                if other_company == company:
                    continue
                position = answer_text.lower().find(
                    other_company.lower(), company_start + len(company)
                )
                if position >= 0:
                    company_end = min(company_end, position)
            checks.append(
                numeric_match(answer_text[company_start:company_end], expected, item.tolerance)
            )
    if item.expected_leader:
        checks.append(leader_match(answer_text, item.expected_leader))
    if item.expected_periods:
        checks.append(keywords_match(answer_text, item.expected_periods))
    correct = all(checks) if checks else False

    allowed = allowed_filenames_for(item)
    required = (
        {COMPANY_FILENAMES[c] for c in item.expected_company_amounts_millions}
        if item.expected_company_amounts_millions
        else set()
    )
    grounded = (
        len(citation_filenames) > 0
        and all(f in allowed for f in citation_filenames)
        and required.issubset(set(citation_filenames))
    )
    text_supported = (
        True
        if citation_texts is None
        else citation_text_supports_expected(item, citation_texts, citation_filenames)
    )
    citation_association_valid = (
        True
        if citation_texts is None or citation_numbers is None
        else citation_company_associations_valid(
            item,
            answer_text,
            citation_filenames,
            citation_texts,
            citation_numbers,
        )
    )
    detail = (
        f"correct={correct} (expected_amount_millions={item.expected_amount_millions}, "
        f"expected_company_amounts_millions={item.expected_company_amounts_millions}, "
        f"expected_leader={item.expected_leader}, expected_periods={item.expected_periods}, "
        f"expected_keywords={item.expected_keywords}), "
        f"grounded={grounded} (citations={citation_filenames}, allowed={sorted(allowed)}), "
        f"text_supported={text_supported}, "
        f"citation_association_valid={citation_association_valid}"
    )
    return ScoreResult(
        correct=correct,
        grounded=grounded,
        text_supported=text_supported,
        detail=detail,
        citation_association_valid=citation_association_valid,
    )
