"""
PDF field extraction using spaCy NER and regex patterns.
Extracts scalar fields (servings, times, category, name, etc.) from PDF text.
"""

from __future__ import annotations

import re
from typing import Optional

from app.ml.pattern_store import PatternStore
from app.ml.feature_extractor import TextBlock


class FieldExtractor:
    """Extract scalar fields from PDF text using spaCy + regex patterns."""

    def __init__(self, pattern_store: PatternStore):
        self.pattern_store = pattern_store
        self.spacy_model = None

        # Try to load spacy model
        try:
            import spacy

            try:
                self.spacy_model = spacy.load("pt_core_news_sm")
            except OSError:
                try:
                    self.spacy_model = spacy.load("es_core_news_sm")
                except OSError:
                    pass  # Will fall back to regex only
        except ImportError:
            pass

    def extract_fields(
        self,
        text_blocks: list[TextBlock],
        restaurant_id: Optional[str] = None,
    ) -> dict:
        """
        Extract scalar fields from text blocks.
        Returns dict of extracted fields.
        """
        full_text = "\n\n".join(b.text for b in text_blocks)
        language = self._detect_language(full_text)

        result = {}

        # Extract each field using patterns
        for field_name in [
            "servings",
            "prep_time_minutes",
            "cooking_time_minutes",
            "category",
            "country",
            "region",
        ]:
            value = self._extract_field(
                field_name,
                full_text,
                language,
                restaurant_id,
            )
            if value is not None:
                result[field_name] = value

        # Extract name from first block (usually title)
        if text_blocks:
            first_text = text_blocks[0].text.strip()
            result["name"] = first_text.split("\n")[0][:100]  # First line, max 100 chars

        return result

    def _detect_language(self, text: str) -> str:
        """Detect language (pt, es, en) from text."""
        try:
            from langdetect import detect

            lang = detect(text[:500])
            if lang in ("pt", "es", "en"):
                return lang
        except Exception:
            pass
        return "pt"  # Default to Portuguese

    def _extract_field(
        self,
        field_name: str,
        text: str,
        language: str,
        restaurant_id: Optional[str] = None,
    ) -> Optional[str | int]:
        """Extract a single field using regex patterns."""
        patterns = self.pattern_store.get_pdf_field_patterns(
            field_name,
            language=language,
            restaurant=restaurant_id,
        )

        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    # Return first match (could be string or tuple)
                    result = matches[0]
                    if isinstance(result, tuple):
                        result = result[0]

                    # Try to convert to int for time/servings fields
                    if field_name in ("servings", "prep_time_minutes", "cooking_time_minutes"):
                        try:
                            return int(result)
                        except (ValueError, TypeError):
                            pass

                    return str(result).strip()
            except re.error:
                continue

        return None

    def extract_ingredients_from_table(
        self,
        headers: list[str],
        rows: list[dict],
        restaurant_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Extract ingredients from tabular data.
        Assumes column_classifier has already mapped headers to canonical names.
        """
        # This would be called from normalizer after column classification
        # Returns list of ingredient dicts
        return rows
