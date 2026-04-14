"""
Local ML normalizer orchestrating all modules.
Replaces ai_normalizer.py for offline, learning-based normalization.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.ml.column_classifier import ColumnClassifier
from app.ml.field_extractor import FieldExtractor, TextBlock
from app.ml.feature_extractor import extract_column_features, extract_pdf_text_blocks
from app.ml.pattern_store import PatternStore
from app.models.schema import Ingredient, NormalizedMenuSheet, PreparationStep


class LocalNormalizer:
    """Local ML normalizer - no external APIs required."""

    def __init__(self, pattern_store: PatternStore, models_dir: Optional[Path] = None):
        self.pattern_store = pattern_store
        self.models_dir = models_dir or Path("storage/ml/models")
        self.column_classifier = ColumnClassifier(pattern_store, self.models_dir)
        self.field_extractor = FieldExtractor(pattern_store)

    async def normalize(
        self,
        raw_content: dict,
        source_file: str,
        source_format: str,
    ) -> NormalizedMenuSheet:
        """
        Normalize a menu sheet using local ML.
        source_format: "pdf", "csv", or "xlsx"
        """
        restaurant_id = self.pattern_store.detect_restaurant_id(source_file)
        self.pattern_store.update_restaurant_profile(restaurant_id)

        if source_format == "pdf":
            return await self._normalize_pdf(raw_content, source_file, restaurant_id)
        elif source_format in ("csv", "xlsx"):
            return await self._normalize_tabular(raw_content, source_file, source_format, restaurant_id)
        else:
            raise ValueError(f"Unsupported format: {source_format}")

    async def _normalize_tabular(
        self,
        raw_content: dict,
        source_file: str,
        source_format: str,
        restaurant_id: str,
    ) -> NormalizedMenuSheet:
        """Normalize CSV/XLSX using column classifier."""
        # Extract headers and rows
        if source_format == "xlsx":
            sheets = raw_content.get("sheets", {})
            # Use first non-empty sheet
            headers = []
            rows = []
            for sheet_name, sheet_data in sheets.items():
                headers = sheet_data.get("headers", [])
                rows = sheet_data.get("rows", [])
                if headers and rows:
                    break
        else:  # csv
            # CSV is already parsed as rows
            raw_text = raw_content.get("raw_text", "")
            lines = raw_text.strip().split("\n")
            if lines:
                headers = lines[0].split(",")
                rows = []
                for line in lines[1:]:
                    row_values = line.split(",")
                    rows.append(dict(zip(headers, row_values)))
            else:
                headers = []
                rows = []

        # Classify columns
        header_mapping = self.column_classifier.classify_headers(headers)

        # Get restaurant overrides
        overrides = self.pattern_store.get_restaurant_column_overrides(restaurant_id)
        for raw_name, canonical in overrides.items():
            header_mapping[raw_name] = canonical

        # Build normalized output
        ingredients = []
        steps = []
        scalar_fields = {}

        for row in rows:
            # Map row to canonical fields
            canonical_row = {}
            for raw_name, value in row.items():
                canonical = header_mapping.get(raw_name, "IGNORE")
                if canonical != "IGNORE":
                    canonical_row[canonical] = value

            # Categorize as ingredient or step
            if "product_name" in canonical_row or "quantity" in canonical_row:
                # Ingredient row
                ingredients.append(self._build_ingredient(canonical_row, restaurant_id))
            elif "action" in canonical_row or "step_number" in canonical_row:
                # Step row
                steps.append(self._build_step(canonical_row))

            # Collect scalar fields (name, category, servings, etc.)
            for key in ["name", "category", "servings", "country", "region", "continent"]:
                if key in canonical_row and key not in scalar_fields:
                    scalar_fields[key] = canonical_row[key]

        # Compute confidence
        n_required = 4
        n_filled = sum(1 for k in ["name", "category", "servings"] if k in scalar_fields and scalar_fields[k])
        field_score = (n_filled / n_required) * 0.6
        optional_fields = ["country", "region", "continent"]
        n_optional_filled = sum(1 for k in optional_fields if k in scalar_fields and scalar_fields[k])
        n_optional_score = (n_optional_filled / len(optional_fields)) * 0.4
        confidence = field_score + n_optional_score

        # Build sheet
        sheet = NormalizedMenuSheet(
            id=source_file.split(".")[0],
            name=scalar_fields.get("name", "Unknown"),
            category=scalar_fields.get("category", "Other"),
            country=scalar_fields.get("country"),
            region=scalar_fields.get("region"),
            continent=scalar_fields.get("continent"),
            servings=self._parse_int(scalar_fields.get("servings", 1)),
            prep_time_minutes=self._parse_int(scalar_fields.get("prep_time_minutes")),
            cooking_time_minutes=self._parse_int(scalar_fields.get("cooking_time_minutes")),
            ingredients=ingredients,
            steps=steps,
            confidence_score=confidence,
            source_file=source_file,
            source_format=source_format,
            normalized_at=datetime.utcnow(),
            raw_extraction=raw_content,
        )

        return sheet

    async def _normalize_pdf(
        self,
        raw_content: dict,
        source_file: str,
        restaurant_id: str,
    ) -> NormalizedMenuSheet:
        """Normalize PDF using field extractor + optional table detection."""
        pdf_text = raw_content.get("full_text", "")
        tables = raw_content.get("tables", [])

        # Extract text blocks
        text_blocks = extract_pdf_text_blocks(pdf_text)

        # Extract scalar fields
        scalar_fields = self.field_extractor.extract_fields(text_blocks, restaurant_id)

        ingredients = []
        steps = []

        # If tables exist, try to parse them
        if tables:
            for table in tables:
                if not table:
                    continue

                headers = table[0]
                rows = table[1:]

                # Classify table columns
                header_mapping = self.column_classifier.classify_headers(headers)

                for row in rows:
                    canonical_row = {}
                    for i, header in enumerate(headers):
                        if i < len(row):
                            canonical = header_mapping.get(header, "IGNORE")
                            if canonical != "IGNORE":
                                canonical_row[canonical] = row[i]

                    if "product_name" in canonical_row or "quantity" in canonical_row:
                        ingredients.append(self._build_ingredient(canonical_row, restaurant_id))
                    elif "action" in canonical_row or "step_number" in canonical_row:
                        steps.append(self._build_step(canonical_row))

        # Confidence calculation
        n_required = 4
        n_filled = sum(1 for k in ["name", "category", "servings"] if k in scalar_fields and scalar_fields[k])
        field_score = (n_filled / n_required) * 0.6
        confidence = field_score

        sheet = NormalizedMenuSheet(
            id=source_file.split(".")[0],
            name=scalar_fields.get("name", "Unknown"),
            category=scalar_fields.get("category", "Other"),
            country=scalar_fields.get("country"),
            region=scalar_fields.get("region"),
            continent=scalar_fields.get("continent"),
            servings=scalar_fields.get("servings", 1),
            prep_time_minutes=scalar_fields.get("prep_time_minutes"),
            cooking_time_minutes=scalar_fields.get("cooking_time_minutes"),
            ingredients=ingredients,
            steps=steps,
            confidence_score=confidence,
            source_file=source_file,
            source_format="pdf",
            normalized_at=datetime.utcnow(),
            raw_extraction=raw_content,
        )

        return sheet

    def _build_ingredient(self, canonical_row: dict, restaurant_id: str) -> Ingredient:
        """Build an Ingredient from normalized row."""
        line_number = self._parse_int(canonical_row.get("line_number", 0))
        product_name = str(canonical_row.get("product_name", "Unknown"))
        quantity = self._parse_float(canonical_row.get("quantity", 0.0))
        unit = self.pattern_store.normalize_unit(
            str(canonical_row.get("unit", "UN")),
            restaurant_id,
        )
        unit_price = self._parse_float(canonical_row.get("unit_price", 0.0))
        observations = canonical_row.get("observations")

        return Ingredient(
            line_number=line_number,
            product_name=product_name,
            quantity=quantity,
            unit=unit,
            unit_price=unit_price,
            observations=observations,
        )

    def _build_step(self, canonical_row: dict) -> PreparationStep:
        """Build a PreparationStep from normalized row."""
        return PreparationStep(
            step_number=self._parse_int(canonical_row.get("step_number", 0)),
            action=str(canonical_row.get("action", "Prepare")),
            product=canonical_row.get("product"),
            cut_type=canonical_row.get("cut_type"),
            time_minutes=self._parse_int(canonical_row.get("time_minutes")),
            observations=canonical_row.get("observations"),
        )

    @staticmethod
    def _parse_int(value) -> Optional[int]:
        """Safely parse integer."""
        if value is None:
            return None
        try:
            return int(float(str(value)))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_float(value) -> float:
        """Safely parse float."""
        try:
            return float(str(value)) if value else 0.0
        except (ValueError, TypeError):
            return 0.0
