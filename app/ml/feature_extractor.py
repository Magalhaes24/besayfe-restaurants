"""
Feature extraction for column classification and field extraction.
Handles both tabular (CSV/XLSX) and text-based (PDF) data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextBlock:
    """Represents a text block from PDF (paragraph or text run)."""

    text: str
    page: int
    is_in_table: bool = False


@dataclass
class ColumnFeatures:
    """Features extracted from a single column header."""

    raw_name: str
    position: int  # 0-indexed column position
    has_currency: bool
    is_all_caps: bool
    has_number: bool
    length: int


def extract_column_features(headers: list[str]) -> list[ColumnFeatures]:
    """
    Extract metadata features from column headers for TF-IDF + metadata vectorization.
    """
    features = []
    for pos, header in enumerate(headers):
        normalized = header.strip().lower() if header else ""

        has_currency = "€" in header or "$" in header or "£" in header
        is_all_caps = header.isupper() if header else False
        has_number = any(c.isdigit() for c in header)
        length = len(normalized)

        features.append(
            ColumnFeatures(
                raw_name=header or "",
                position=pos,
                has_currency=has_currency,
                is_all_caps=is_all_caps,
                has_number=has_number,
                length=length,
            )
        )

    return features


def extract_pdf_text_blocks(pdf_text: str, is_in_table: bool = False) -> list[TextBlock]:
    """
    Parse PDF text into logical text blocks (paragraphs).
    """
    blocks = []
    page = 1

    # Simple paragraph splitting by double newlines
    paragraphs = pdf_text.split("\n\n")
    for para in paragraphs:
        para = para.strip()
        if para:
            blocks.append(
                TextBlock(
                    text=para,
                    page=page,
                    is_in_table=is_in_table,
                )
            )

    return blocks
