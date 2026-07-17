from sage.generation import prompts
from sage.retrieval.retriever import RetrievedChunk


def _chunk(chunk_id: int, company: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        chunk_index=0,
        text=f"{company} chunk text",
        page_number=1,
        filename="f.pdf",
        company=company,
        fiscal_year="FY24",
        doc_type="10-K",
        score=1.0,
    )


def test_is_comparison_false_for_single_company():
    assert prompts.is_comparison([_chunk(1, "Apple"), _chunk(2, "Apple")]) is False


def test_is_comparison_true_for_multiple_companies():
    assert prompts.is_comparison([_chunk(1, "Apple"), _chunk(2, "Microsoft")]) is True


def test_is_comparison_false_for_no_chunks():
    assert prompts.is_comparison([]) is False


def test_build_messages_selects_comparison_system_prompt_for_multi_company():
    messages = prompts.build_messages("compare capex", [_chunk(1, "Apple"), _chunk(2, "Microsoft")])

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == prompts.COMPARISON_SYSTEM_PROMPT
    assert "one clearly labeled section per company" in messages[0]["content"]


def test_build_messages_selects_plain_system_prompt_for_single_company():
    messages = prompts.build_messages("what were margins", [_chunk(1, "Apple")])

    assert messages[0]["content"] == prompts.SYSTEM_PROMPT


def test_build_messages_inserts_history_between_system_and_new_user_message():
    history = [
        {"role": "user", "content": "prior question"},
        {"role": "assistant", "content": "prior answer"},
    ]

    messages = prompts.build_messages("follow-up question", [_chunk(1, "Apple")], history=history)

    assert messages[0]["role"] == "system"
    assert messages[1] == history[0]
    assert messages[2] == history[1]
    assert messages[3]["role"] == "user"
    assert "follow-up question" in messages[3]["content"]


def test_build_context_block_includes_company_header_and_text():
    block = prompts.build_context_block([_chunk(1, "Apple")])

    assert "Company: Apple" in block
    assert "chunk_id=1" in block
    assert "Apple chunk text" in block


def test_citation_format_instruction_present_in_user_message():
    message = prompts.build_user_message("q", [_chunk(1, "Apple")])
    assert "```citations" in message
