"""CLI entrypoints for Sage (ingest, ask, conversations).

This is the primary proof-of-correctness for the whole backend/CLI layer
before any HTTP API or frontend exists -- every module the pipeline touches
(ingest, retrieval, reranking, generation, caching, conversation history) is
reachable from here.
"""

import argparse
from pathlib import Path

from config import settings
from sage.db.conversations import (
    append_message,
    create_conversation,
    get_history,
    list_conversations,
)
from sage.db.database import init_db
from sage.generation.answer_engine import generate_answer, generate_answer_stream
from sage.ingest.pipeline import ingest_folder


def cmd_ingest(args: argparse.Namespace) -> None:
    init_db()
    input_dir = Path(args.input_dir)
    documents = ingest_folder(input_dir)
    if not documents:
        print(f"No PDFs found in {input_dir}")
        return

    for document in documents:
        print(
            f"{document.filename}: {document.page_count} pages, "
            f"company={document.company} fiscal_year={document.fiscal_year} "
            f"doc_type={document.doc_type} -> status={document.status}"
        )


def _resolve_conversation(args: argparse.Namespace) -> int | None:
    """Returns a conversation id to continue/persist to, or None if the ask
    is a one-off (no --conversation-id and no --new-conversation given)."""
    if args.conversation_id is not None:
        return args.conversation_id
    if args.new_conversation is not None:
        title = args.new_conversation or args.text[:60]
        conversation_id, _session_token = create_conversation(title=title)
        print(f"Started conversation #{conversation_id}: {title}")
        return conversation_id
    return None


def cmd_ask(args: argparse.Namespace) -> None:
    init_db()
    companies = args.company or None
    conversation_id = _resolve_conversation(args)
    history = get_history(conversation_id) if conversation_id is not None else None

    if companies and len(companies) > 1:
        print(f"Comparison mode: {', '.join(companies)}")

    if args.stream:
        result = None
        for item in generate_answer_stream(
            args.text, top_k=args.top_k, companies=companies, history=history
        ):
            if isinstance(item, str):
                print(item, end="", flush=True)
            else:
                result = item
                print()
                print()
                _print_citations(result.citations)
                _print_stats(result)
    else:
        result = generate_answer(args.text, top_k=args.top_k, companies=companies, history=history)
        print(result.answer_text)
        print()
        _print_citations(result.citations)
        _print_stats(result)

    if conversation_id is not None and result is not None:
        append_message(conversation_id, "user", args.text)
        append_message(
            conversation_id,
            "assistant",
            result.answer_text,
            citations=[
                {
                    "n": c.n,
                    "chunk_id": c.chunk_id,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "company": c.company,
                }
                for c in result.citations
            ],
        )


def cmd_conversations(args: argparse.Namespace) -> None:
    init_db()
    conversations = list_conversations()
    if not conversations:
        print("No conversations yet.")
        return
    for c in conversations:
        print(f"#{c.id}  {c.title or '(untitled)'}  created={c.created_at}")


def _print_citations(citations) -> None:
    if not citations:
        print("(no citations resolved)")
        return
    print("Citations:")
    for c in citations:
        print(
            f"  [{c.n}] {c.filename} p.{c.page_number} ({c.company} {c.fiscal_year}) "
            f"— chunk_id={c.chunk_id}"
        )


def _print_stats(result) -> None:
    print(
        f"\nmodel={result.model} retrieval={result.retrieval_latency_ms:.0f}ms "
        f"generation={result.generation_latency_ms:.0f}ms total={result.total_latency_ms:.0f}ms "
        f"tokens(prompt={result.prompt_tokens}, completion={result.completion_tokens}) "
        f"cache_hit={result.cache_hit} cost_usd={result.cost_usd:.6f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sage")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest, chunk, embed, and store PDFs from a folder"
    )
    ingest_parser.add_argument("--input-dir", default=str(settings.RAW_DIR))
    ingest_parser.set_defaults(func=cmd_ingest)

    ask_parser = subparsers.add_parser(
        "ask", help="Retrieve + generate a grounded, cited answer (repeat --company to compare)"
    )
    ask_parser.add_argument("text")
    ask_parser.add_argument("--top-k", type=int, default=settings.DEFAULT_TOP_K)
    ask_parser.add_argument(
        "--company",
        action="append",
        default=None,
        help="Restrict to one company; repeat to compare multiple companies in one answer",
    )
    ask_parser.add_argument("--fiscal-year", default=None)
    ask_parser.add_argument("--doc-type", default=None)
    ask_parser.add_argument("--stream", action="store_true")
    ask_parser.add_argument(
        "--conversation-id", type=int, default=None, help="Continue an existing conversation"
    )
    ask_parser.add_argument(
        "--new-conversation",
        nargs="?",
        const="",
        default=None,
        help="Start a new conversation (optionally with a title) and persist this turn to it",
    )
    ask_parser.set_defaults(func=cmd_ask)

    conversations_parser = subparsers.add_parser("conversations", help="List past conversations")
    conversations_parser.set_defaults(func=cmd_conversations)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
