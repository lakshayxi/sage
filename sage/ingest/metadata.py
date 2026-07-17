"""Filename-convention metadata parsing + manual override hook.

Expected convention: "<Company>_<FiscalYear>_<doc_type>.pdf", e.g.
"Apple_FY24_10-K.pdf" -> company=Apple, fiscal_year=FY24, doc_type=10-K.

Ported unchanged from the reference project's `finresearch/ingest/metadata.py`
-- the parsing rules aren't provider- or product-specific.
"""

import re
from dataclasses import dataclass, replace
from pathlib import Path

KNOWN_DOC_TYPES = ["10-K", "10-Q", "annual_report", "transcript", "filing", "presentation", "news"]
FISCAL_YEAR_RE = re.compile(r"^FY\d{2,4}$", re.IGNORECASE)


@dataclass
class DocumentMetadata:
    company: str
    fiscal_year: str | None
    doc_type: str


def parse_filename_metadata(filename: str) -> DocumentMetadata:
    """Best-effort parse of company/fiscal_year/doc_type from a filename."""
    stem = Path(filename).stem
    parts = [p for p in stem.split("_") if p]

    if not parts:
        return DocumentMetadata(company="unknown", fiscal_year=None, doc_type="unknown")

    company = parts[0]
    rest = parts[1:]

    fiscal_year = None
    if rest and FISCAL_YEAR_RE.match(rest[0]):
        fiscal_year = rest[0].upper()
        rest = rest[1:]

    remainder = "_".join(rest).lower()
    doc_type = next((t for t in KNOWN_DOC_TYPES if t.lower() == remainder), None)
    if doc_type is None:
        doc_type = next(
            (t for t in KNOWN_DOC_TYPES if t.lower() in remainder), remainder or "unknown"
        )

    return DocumentMetadata(company=company, fiscal_year=fiscal_year, doc_type=doc_type)


def apply_overrides(metadata: DocumentMetadata, **overrides) -> DocumentMetadata:
    """Apply manual overrides (e.g. from an upload form) on top of parsed metadata.

    Only non-None override values are applied.
    """
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(metadata, **clean)
