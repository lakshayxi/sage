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

Requires a live `GEMINI_API_KEY` (`.env` at the repo root) -- there is no
stub/offline mode, unlike `pytest tests/`. Every item costs one real Gemini
chat call (plus a local embedding + rerank pass, $0). At 14 items this is a
small, deliberate spend against a free-tier key already documented
(`docs/llm-engineer-work-log.md`) to have a hard, non-recoverable quota
wall on *embeddings* (now local, so moot) -- chat quota headroom is less
well characterized, so this harness is meant to be run occasionally to spot
real regressions, not in a tuning loop. `--limit N` runs a faster/cheaper
subset for a first pass.

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
from eval.dataset import EVAL_ITEMS, EvalItem
from eval.scoring import score_item
from sage.db.database import init_db
from sage.generation.answer_engine import generate_answer

RESULTS_DIR = settings.BASE_DIR / "eval" / "results"

# Courtesy pause between live Gemini calls -- not a documented rate limit
# requirement, just cheap insurance against bursting a free-tier per-minute
# cap given this project's history of quota surprises (see module docstring).
_PAUSE_BETWEEN_ITEMS_SECONDS = 1.0


def _run_item(item: EvalItem) -> dict:
    start = time.perf_counter()
    result = generate_answer(
        item.question,
        companies=item.companies,
        fiscal_year=item.fiscal_year,
        doc_type=item.doc_type,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    citation_filenames = [c.filename for c in result.citations]
    score = score_item(item, result.answer_text, citation_filenames)

    return {
        "id": item.id,
        "question": item.question,
        "answerable": item.answerable,
        "expected_answer": item.expected_answer,
        "answer": result.answer_text,
        "citation_filenames": ";".join(citation_filenames),
        "model": result.model,
        "cache_hit": result.cache_hit,
        "cost_usd": round(result.cost_usd, 6),
        "latency_ms": round(latency_ms, 1),
        "correct": score.correct,
        "grounded": score.grounded,
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
        f"Cache hits: {cache_hits}/{len(rows)}",
        f"Total estimated cost: ${total_cost:.6f}",
        "",
        "## Per-item results",
        "",
        "| id | answerable | passed | correct | grounded | latency_ms | cache_hit |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['answerable']} | {r['passed']} | {r['correct']} | "
            f"{r['grounded']} | {r['latency_ms']} | {r['cache_hit']} |"
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
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    init_db()

    items = EVAL_ITEMS[: args.limit] if args.limit is not None else EVAL_ITEMS

    rows: list[dict] = []
    start = time.perf_counter()
    for i, item in enumerate(items, start=1):
        print(f"[{i}/{len(items)}] {item.id}: {item.question}")
        row = _run_item(item)
        rows.append(row)
        status = "PASS" if row["passed"] else "FAIL"
        print(f"    {status}  correct={row['correct']} grounded={row['grounded']}")
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
