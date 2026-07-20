"""Evaluation harness entry point.

Runs the hand-curated Q&A set (`eval/dataset.py`) through the real pipeline
(`sage.generation.answer_engine.generate_answer`, live Gemini) against
Sage's actual ingested corpus (`data/raw/*.pdf` -> `db/sage.db` +
`data/chroma`) and scores each answer with the deterministic heuristics in
`eval/scoring.py` -- see that module's docstring for why a heuristic was
chosen over an LLM-judge here (short version: no free/unmetered local judge
exists for a Gemini-only project, this dataset has objectively-correct
numeric answers a heuristic checks more reliably than a judge model would,
and it avoids a heavy eval-framework dependency for a harness explicitly
scoped as basic/correctness-only).

Every run passes `use_cache=False` to `generate_answer` -- without it, a
second run against the same dataset would silently replay whatever the
*first* run generated (via the exact-match/semantic query cache) instead of
actually re-testing generation, making this harness useless for catching a
same-day prompt/retrieval regression. This does mean every run spends full
retrieval + generation cost/latency on every item, every time, by design.

Beyond the original correct/grounded scoring, each item also reports:
- **recall@k**: whether hybrid retrieval (before reranking/selection)
  surfaced gold-value evidence for every expected comparison company (or
  the expected value/source for a single-company item).
- **gate_passed**: legacy field name for whether evidence selection returned
  anything. Ordinary queries use `MIN_RERANK_SCORE`; explicit comparisons
  use requested-company completeness plus bounded per-company rank selection.
- **citation_mapping_valid**: whether every resolved citation's `n`
  deterministically points at the right retrieved chunk
  (`retrieved_chunk_ids[n - 1]`) -- an end-to-end live check of the
  citation-integrity fix in `sage/generation/answer_engine.py`.
- **text_supported**: whether the *text* of the cited chunk(s), not just
  their filename, actually contains/supports the expected value (see
  `eval/scoring.py`'s `citation_text_supports_expected`).

Requires a live `GEMINI_API_KEY` (`.env` at the repo root) -- there is no
stub/offline mode, unlike `pytest tests/`. Every item costs one real Gemini
chat call (plus a local embedding + rerank pass, $0, and one extra local
retrieval-only call for recall@k, also $0). At ~19 items this is a small,
deliberate spend against a free-tier key already documented
(`docs/llm-engineer-work-log.md`) to have a hard, non-recoverable quota
wall on *embeddings* (now local, so moot) -- chat quota headroom is less
well characterized, so this harness is meant to be run occasionally to spot
real regressions, not in a tuning loop. `--limit N` runs a faster/cheaper
subset for a first pass; repeat `--id ITEM_ID` for a targeted subset.

Usage:
    .venv/bin/python -m eval.run_eval [--limit N] [--out-dir DIR]

or, once installed:
    .venv/bin/sage-eval [--limit N] [--out-dir DIR]
"""

import argparse
import csv
import time
from datetime import UTC, datetime
from pathlib import Path

from config import settings
from eval.dataset import COMPANY_FILENAMES, EVAL_ITEMS, EvalItem
from eval.scoring import allowed_filenames_for, score_item, source_text_contains_amount
from sage.db.conversations import HistoryTurn
from sage.db.database import init_db
from sage.generation.answer_engine import generate_answer
from sage.retrieval.retriever import retrieve_hybrid

RESULTS_DIR = settings.BASE_DIR / "eval" / "results"

# Courtesy pause between live Gemini calls -- not a documented rate limit
# requirement, just cheap insurance against bursting a free-tier per-minute
# cap given this project's history of quota surprises (see module docstring).
_PAUSE_BETWEEN_ITEMS_SECONDS = 1.0


def _item_history(item: EvalItem) -> list[HistoryTurn] | None:
    if not item.history:
        return None
    return [HistoryTurn(role=h["role"], content=h["content"]) for h in item.history]


def _recall_at_k(item: EvalItem) -> bool | None:
    """Whether hybrid retrieval alone (before reranking/gating) surfaced at
    least one chunk from the question's expected source document, at the
    same candidate width (`RERANK_CANDIDATE_K`) `generate_answer` uses
    internally. None (not applicable) for a deliberately unanswerable item
    -- there's no "expected source" to recall."""
    if not item.answerable:
        return None
    allowed = allowed_filenames_for(item)
    candidates = retrieve_hybrid(
        item.question,
        top_k=settings.RERANK_CANDIDATE_K,
        companies=item.companies,
        fiscal_year=item.fiscal_year,
        doc_type=item.doc_type,
    )
    if item.expected_company_amounts_millions:
        return all(
            any(
                c.filename == COMPANY_FILENAMES[company]
                and source_text_contains_amount(c.text, expected)
                for c in candidates
            )
            for company, expected in item.expected_company_amounts_millions.items()
        )
    if item.expected_amount_millions is not None:
        return any(
            c.filename in allowed
            and source_text_contains_amount(c.text, item.expected_amount_millions)
            for c in candidates
        )
    return any(c.filename in allowed for c in candidates)


def _citation_mapping_valid(retrieved_chunk_ids: list[int], citations) -> bool:
    """Live, end-to-end check of the citation-integrity fix: every resolved
    citation's `n` must deterministically resolve to
    `retrieved_chunk_ids[n - 1]` -- the exact chunk shown to the model as
    "[n]" -- with nothing out of range or mismatched."""
    for c in citations:
        if not (1 <= c.n <= len(retrieved_chunk_ids)):
            return False
        if retrieved_chunk_ids[c.n - 1] != c.chunk_id:
            return False
    return True


def _run_item(item: EvalItem) -> dict:
    recall = _recall_at_k(item)

    start = time.perf_counter()
    result = generate_answer(
        item.question,
        companies=item.companies,
        fiscal_year=item.fiscal_year,
        doc_type=item.doc_type,
        history=_item_history(item),
        use_cache=False,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    citation_filenames = [c.filename for c in result.citations]
    citation_texts = [c.text for c in result.citations]
    score = score_item(item, result.answer_text, citation_filenames, citation_texts)
    gate_passed = len(result.retrieved_chunk_ids) > 0
    citation_mapping_valid = _citation_mapping_valid(result.retrieved_chunk_ids, result.citations)

    return {
        "id": item.id,
        "question": item.question,
        "answerable": item.answerable,
        "multi_turn": item.history is not None,
        "expected_answer": item.expected_answer,
        "answer": result.answer_text,
        "citation_filenames": ";".join(citation_filenames),
        "model": result.model,
        "cache_hit": result.cache_hit,
        "cost_usd": round(result.cost_usd, 6),
        "latency_ms": round(latency_ms, 1),
        "recall_at_k": recall,
        "gate_passed": gate_passed,
        "citation_mapping_valid": citation_mapping_valid,
        "correct": score.correct,
        "grounded": score.grounded,
        "text_supported": score.text_supported,
        "passed": score.passed,
        "detail": score.detail,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _rate(rows: list[dict], key: str) -> str:
    """`n/d passed` string over rows where `key` isn't None (N/A)."""
    applicable = [r for r in rows if r[key] is not None]
    if not applicable:
        return "n/a"
    passed = sum(1 for r in applicable if r[key])
    return f"{passed}/{len(applicable)}"


def _write_markdown(path: Path, rows: list[dict], total_elapsed_s: float) -> None:
    passed = sum(1 for r in rows if r["passed"])
    total_cost = sum(r["cost_usd"] for r in rows)
    cache_hits = sum(1 for r in rows if r["cache_hit"])

    lines = [
        "# Sage -- Eval Report",
        "",
        f"Items evaluated: {len(rows)} (of {len(EVAL_ITEMS)} in the full dataset)",
        f"Pass rate: {passed}/{len(rows)}",
        f"Total wall time: {total_elapsed_s:.1f}s",
        f"Cache hits: {cache_hits}/{len(rows)} (expected 0 -- use_cache=False every run)",
        f"Total estimated cost: ${total_cost:.6f}",
        "",
        "## Aggregate retrieval/citation metrics",
        "",
        f"- recall@k (answerable items only): {_rate(rows, 'recall_at_k')}",
        f"- evidence selection passed: {_rate(rows, 'gate_passed')}",
        f"- citation number->chunk mapping valid: {_rate(rows, 'citation_mapping_valid')}",
        f"- citation text supports expected value: {_rate(rows, 'text_supported')}",
        "",
        "## Per-item results",
        "",
        "| id | answerable | multi_turn | passed | correct | grounded | text_supported | "
        "recall_at_k | gate_passed | citation_mapping_valid | latency_ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['answerable']} | {r['multi_turn']} | {r['passed']} | "
            f"{r['correct']} | {r['grounded']} | {r['text_supported']} | "
            f"{r['recall_at_k']} | {r['gate_passed']} | {r['citation_mapping_valid']} | "
            f"{r['latency_ms']} |"
        )
    lines.append("")
    lines.append("## Failure detail")
    lines.append("")
    failed = [r for r in rows if not r["passed"]]
    if not failed:
        lines.append("(none -- all items passed)")
    for r in failed:
        lines.append(f"- **{r['id']}**: {r['detail']}")
        lines.append(f"  - question: {r['question']}")
        lines.append(f"  - expected: {r['expected_answer']}")
        lines.append(f"  - answer: {r['answer']}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Sage eval harness against the live RAG pipeline."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"Evaluate only the first N items (of {len(EVAL_ITEMS)} total).",
    )
    parser.add_argument(
        "--id",
        action="append",
        dest="item_ids",
        choices=[item.id for item in EVAL_ITEMS],
        help="Evaluate one item by id; repeat to run a targeted subset.",
    )
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    init_db()

    if args.item_ids:
        requested_ids = set(args.item_ids)
        items = [item for item in EVAL_ITEMS if item.id in requested_ids]
    else:
        items = EVAL_ITEMS[: args.limit] if args.limit is not None else EVAL_ITEMS

    rows: list[dict] = []
    start = time.perf_counter()
    for i, item in enumerate(items, start=1):
        print(f"[{i}/{len(items)}] {item.id}: {item.question}")
        row = _run_item(item)
        rows.append(row)
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"    {status}  correct={row['correct']} grounded={row['grounded']} "
            f"text_supported={row['text_supported']} recall_at_k={row['recall_at_k']} "
            f"citation_mapping_valid={row['citation_mapping_valid']}"
        )
        if i < len(items):
            time.sleep(_PAUSE_BETWEEN_ITEMS_SECONDS)
    total_elapsed_s = time.perf_counter() - start

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"eval_{timestamp}.csv"
    md_path = out_dir / f"eval_{timestamp}.md"

    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows, total_elapsed_s)

    passed = sum(1 for r in rows if r["passed"])
    print(f"\n{passed}/{len(rows)} passed. Wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
