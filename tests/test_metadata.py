from sage.ingest.metadata import DocumentMetadata, apply_overrides, parse_filename_metadata


def test_parses_company_fiscal_year_doc_type():
    meta = parse_filename_metadata("Apple_FY24_10-K.pdf")
    assert meta.company == "Apple"
    assert meta.fiscal_year == "FY24"
    assert meta.doc_type == "10-K"


def test_fiscal_year_is_case_normalized_upper():
    meta = parse_filename_metadata("Microsoft_fy23_annual_report.pdf")
    assert meta.fiscal_year == "FY23"


def test_missing_fiscal_year_falls_back_to_none():
    meta = parse_filename_metadata("Tesla_transcript.pdf")
    assert meta.company == "Tesla"
    assert meta.fiscal_year is None
    assert meta.doc_type == "transcript"


def test_unknown_doc_type_falls_back_to_remainder():
    meta = parse_filename_metadata("Nvidia_FY25_earnings_call.pdf")
    assert meta.company == "Nvidia"
    assert meta.fiscal_year == "FY25"
    assert meta.doc_type == "earnings_call"


def test_empty_filename_falls_back_to_unknown():
    # An all-underscore stem splits into zero non-empty parts.
    meta = parse_filename_metadata("___.pdf")
    assert meta.company == "unknown"
    assert meta.fiscal_year is None
    assert meta.doc_type == "unknown"


def test_apply_overrides_only_replaces_non_none_values():
    base = DocumentMetadata(company="Apple", fiscal_year="FY24", doc_type="10-K")
    result = apply_overrides(base, company="Apple Inc.", fiscal_year=None, doc_type=None)
    assert result.company == "Apple Inc."
    assert result.fiscal_year == "FY24"
    assert result.doc_type == "10-K"
