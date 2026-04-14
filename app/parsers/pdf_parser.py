from __future__ import annotations

import io
from typing import TypedDict


class RawPDFExtraction(TypedDict):
    full_text: str
    tables: list[list[list[str | None]]]
    page_count: int
    metadata: dict


async def extract_pdf(file_bytes: bytes) -> RawPDFExtraction:
    """
    Try pdfplumber first (better table detection), fall back to PyPDF2.
    Returns a unified RawPDFExtraction dict to send to the AI normalizer.
    """
    try:
        return await _extract_with_pdfplumber(file_bytes)
    except Exception:
        return await _extract_with_pypdf2(file_bytes)


async def _extract_with_pdfplumber(file_bytes: bytes) -> RawPDFExtraction:
    import pdfplumber

    full_text_parts: list[str] = []
    all_tables: list[list[list[str | None]]] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        page_count = len(pdf.pages)
        metadata = pdf.metadata or {}

        for page in pdf.pages:
            # extract_text() returns None if empty
            text = page.extract_text() or ""
            full_text_parts.append(text)

            # extract_tables() returns list of tables
            tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 3,
                    "min_words_vertical": 3,
                    "min_words_horizontal": 1,
                }
            )
            all_tables.extend(tables or [])

    return RawPDFExtraction(
        full_text="\n\n".join(full_text_parts).strip(),
        tables=all_tables,
        page_count=page_count,
        metadata=metadata,
    )


async def _extract_with_pypdf2(file_bytes: bytes) -> RawPDFExtraction:
    import PyPDF2

    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    texts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            texts.append(t)

    meta = {}
    if reader.metadata:
        meta = {k.lstrip("/"): v for k, v in reader.metadata.items()}

    return RawPDFExtraction(
        full_text="\n".join(texts).strip(),
        tables=[],
        page_count=len(reader.pages),
        metadata=meta,
    )
