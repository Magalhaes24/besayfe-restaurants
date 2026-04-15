"""
Local ML normalizer orchestrating all modules.
Replaces ai_normalizer.py for offline, learning-based normalization.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.ml.allergen_detector import AllergenDatabase, IngredientAnalyzer, assess_allergen_risk
from app.ml.column_classifier import ColumnClassifier
from app.ml.field_extractor import FieldExtractor, TextBlock
from app.ml.feature_extractor import extract_column_features, extract_pdf_text_blocks
from app.ml.pattern_store import PatternStore
from app.ml.rule_engine import RuleEngine
from app.models.schema import Ingredient, NormalizedMenuSheet, PreparationStep


class LocalNormalizer:
    """Local ML normalizer - no external APIs required."""

    def __init__(self, pattern_store: PatternStore, models_dir: Optional[Path] = None):
        self.pattern_store = pattern_store
        self.models_dir = models_dir or Path("storage/ml/models")
        self.column_classifier = ColumnClassifier(pattern_store, self.models_dir)
        self.field_extractor = FieldExtractor(pattern_store)
        self.allergen_db = AllergenDatabase()
        self.ingredient_analyzer = IngredientAnalyzer()
        self.rule_engine = RuleEngine(pattern_store)  # Self-learning rule engine

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
        """Normalize CSV/XLSX using column classifier, with fallback to text extraction."""
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

        # If structured extraction yielded no results or looks malformed, try text-based extraction
        # (handles specification sheets, forms, and other non-tabular formats)
        # Malformed detection: headers that look like data values (e.g., "FT-RP-016", numbers, etc.)
        is_malformed = self._is_malformed_tabular_data(headers, rows)
        if not rows or not headers or is_malformed:
            return await self._normalize_tabular_as_text(raw_content, source_file, source_format, restaurant_id)

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
        ing_counter = 0
        step_counter = 0

        for row in rows:
            # Map row to canonical fields
            canonical_row = {}
            for raw_name, value in row.items():
                canonical = header_mapping.get(raw_name, "IGNORE")
                if canonical != "IGNORE":
                    canonical_row[canonical] = value

            # Skip rows with no useful data
            product_name = str(canonical_row.get("product_name", "")).strip() if "product_name" in canonical_row else ""
            action = str(canonical_row.get("action", "")).strip() if "action" in canonical_row else ""

            # Categorize as ingredient or step
            # Prioritize: if it looks like an action, treat as step (even if in product_name field)
            if action or self._is_step_action(product_name):
                # This is a step row
                step_counter += 1
                steps.append(self._build_step(canonical_row, step_counter))
            elif product_name and len(product_name) > 1 and not self._is_step_action(product_name):
                # Check if this is a compound ingredient (multiple ingredients in one cell)
                compound_ingredients = self._parse_compound_ingredient(product_name)
                if compound_ingredients:
                    # Multiple ingredients detected - create separate entries
                    for compound_name, compound_qty, compound_unit in compound_ingredients:
                        ing_counter += 1
                        # Create a modified row for each compound ingredient
                        compound_row = canonical_row.copy()
                        compound_row["product_name"] = compound_name
                        compound_row["quantity"] = compound_qty
                        compound_row["unit"] = compound_unit
                        ingredients.append(self._build_ingredient(compound_row, restaurant_id, ing_counter))
                else:
                    # Single ingredient row
                    ing_counter += 1
                    ingredients.append(self._build_ingredient(canonical_row, restaurant_id, ing_counter))

            # Collect scalar fields (name, category, servings, etc.)
            for key in ["name", "category", "servings", "country", "region", "continent"]:
                if key in canonical_row and key not in scalar_fields:
                    scalar_fields[key] = canonical_row[key]

        # Apply learned rules to enhance ingredient detection
        ingredients = self.rule_engine.apply_learned_rules(ingredients, language="pt")

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

    def _is_malformed_tabular_data(self, headers: list, rows: list) -> bool:
        """
        Detect if extracted tabular data looks malformed.
        E.g., specification sheets where first row values become headers.
        """
        if not headers or len(headers) < 2:
            return False  # Not enough columns to determine

        # Check if headers look like they should be data values
        # (e.g., numeric codes, model numbers, etc.)
        suspicious_headers = 0
        for h in headers[:2]:  # Check first 2 headers
            if h is None:
                continue
            h_str = str(h).strip()
            # If header is all-caps (including multi-word with spaces), or numeric, it's suspicious
            # Normal column headers are typically Title Case or lowercase with underscores
            if h_str and (
                h_str.isupper() and len(h_str) > 3 or  # All caps like "FICHA TECNICA OPERACIONAL"
                re.match(r'^[A-Z0-9\-\.]+$', h_str)     # Codes like "FT-RP-016"
            ):
                suspicious_headers += 1

        # If any headers look like data values, the extraction is likely malformed
        return suspicious_headers > 0

    async def _normalize_tabular_as_text(
        self,
        raw_content: dict,
        source_file: str,
        source_format: str,
        restaurant_id: str,
    ) -> NormalizedMenuSheet:
        """
        Fallback: normalize XLSX/CSV as text when structured extraction fails.
        Handles specification sheets (forms with metadata, ingredients, steps sections).
        """
        raw_text = raw_content.get("raw_text", "")
        if not raw_text:
            # If no raw text, use sheets as fallback (may result in empty sheet)
            raw_text = ""

        # Extract ingredients and steps by looking for section headers
        ingredients = []
        steps = []
        scalar_fields = {}
        ing_counter = 0
        step_counter = 0

        # Split text into lines for processing
        lines = raw_text.split("\n")

        # Track which section we're in
        current_section = "metadata"
        current_table_headers = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect section headers first (must be standalone, not just metadata pairs)
            first_part = line.split("|")[0].strip() if "|" in line else line

            if "===" in line or (first_part.isupper() and len(first_part) > 5):
                # Potential section header
                if "INGREDIENTE" in first_part:
                    current_section = "ingredients"
                    current_table_headers = None
                    continue
                elif "MODO" in first_part or "PASSO" in first_part or "PREPARO" in first_part:
                    current_section = "steps"
                    current_table_headers = None
                    continue
                elif "ALERGEN" in first_part:
                    current_section = "allergens"
                    continue
                elif "===" in line:
                    # Skip sheet markers
                    continue

            # Handle pipe-separated rows
            if "|" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]

                if not parts or len(parts) < 2:
                    # Skip rows with insufficient data
                    if current_section == "metadata":
                        # Try to extract single value as metadata
                        key = first_part.lower()
                        if parts:
                            scalar_fields["category"] = parts[0]
                    continue

                if current_section == "ingredients":
                    # Check if this is a header row
                    parts_lower = [p.lower() for p in parts]
                    is_header_row = any(kw in " ".join(parts_lower) for kw in ["ingrediente", "quantidade", "ingredient", "quantity"])

                    if is_header_row:
                        current_table_headers = parts
                    elif current_table_headers and len(parts) >= 2:
                        # This is a data row
                        ing_counter += 1
                        canonical_row = {}
                        raw_quantity = None

                        for i, header in enumerate(current_table_headers):
                            if i < len(parts):
                                canonical = self.column_classifier.classify_single(header.lower())
                                if canonical and canonical != "IGNORE":
                                    canonical_row[canonical] = parts[i]
                                if header.lower() in ["quantidade", "quantity", "amount"]:
                                    raw_quantity = parts[i]

                        # Parse quantity and unit
                        if raw_quantity:
                            qty, unit = self._parse_quantity_and_unit(raw_quantity)
                            canonical_row["quantity"] = qty
                            canonical_row["unit"] = unit

                        if "product_name" in canonical_row:
                            ingredients.append(self._build_ingredient(canonical_row, restaurant_id, ing_counter))

                elif current_section == "steps":
                    # Check if this is a header row
                    parts_lower = [p.lower() for p in parts]
                    is_header_row = any(kw in " ".join(parts_lower) for kw in ["no.", "passo", "instrucao", "instruction", "step", "action"])

                    if is_header_row:
                        current_table_headers = parts
                    elif current_table_headers and len(parts) >= 2:
                        # This is a data row
                        step_counter += 1
                        canonical_row = {}
                        raw_instruction = None

                        for i, header in enumerate(current_table_headers):
                            if i < len(parts):
                                canonical = self.column_classifier.classify_single(header.lower())
                                if canonical and canonical != "IGNORE":
                                    canonical_row[canonical] = parts[i]
                                if header.lower() in ["instrucao", "instruction", "passo", "step", "action"]:
                                    raw_instruction = parts[i]

                        if raw_instruction and "action" not in canonical_row:
                            canonical_row["action"] = raw_instruction

                        steps.append(self._build_step(canonical_row, step_counter))

                elif current_section == "metadata" and len(parts) >= 2:
                    # Metadata key-value pairs
                    key = parts[0].lower()
                    value = parts[1]

                    if any(k in key for k in ["nome", "name", "titulo", "title"]):
                        scalar_fields["name"] = value
                    elif any(k in key for k in ["codigo", "code", "id"]):
                        scalar_fields["id"] = value
                    elif any(k in key for k in ["categoria", "category"]):
                        scalar_fields["category"] = value
                    elif any(k in key for k in ["pais", "country"]):
                        scalar_fields["country"] = value
                    elif any(k in key for k in ["regiao", "region"]):
                        scalar_fields["region"] = value
                    elif any(k in key for k in ["continente", "continent"]):
                        scalar_fields["continent"] = value
                    elif any(k in key for k in ["porcao", "servings", "pessoas"]):
                        scalar_fields["servings"] = self._parse_int(value)

        # Extract allergens from the metadata text (usually at the end)
        allergens_section = raw_text.lower().split("alergeni")[1] if "alergeni" in raw_text.lower() else ""
        if allergens_section:
            # Extract allergen text from the line (remove pipe separators and clean up)
            allergen_line = allergens_section.split("\n")[0]
            # Remove pipe and anything before it
            if "|" in allergen_line:
                allergen_line = allergen_line.split("|", 1)[1]

            allergen_text = allergen_line.strip()

            # Parse allergens from comma/dash separated list
            # Common allergens from the document: "Gluten - Leite" or "Gluten, Leite"
            allergen_list = re.split(r'[-,]', allergen_text)
            allergen_list = [a.strip().lower() for a in allergen_list if a.strip()]

            # Map common allergen names
            allergen_map = {
                "gluten": "gluten",
                "leite": "milk",
                "lactose": "milk",
                "dairy": "milk",
                "egg": "eggs",
                "ovos": "eggs",
                "fish": "fish",
                "peixe": "fish",
                "shellfish": "shellfish",
                "crustaceo": "shellfish",
                "mollusco": "shellfish",
                "nuts": "nuts",
                "amendoins": "peanuts",
                "peanut": "peanuts",
                "soy": "soy",
                "soja": "soy",
                "sesame": "sesame",
                "sementes de sesamo": "sesame",
            }

            # For each ingredient, mark with detected allergens
            mapped_allergens = [allergen_map.get(a, a) for a in allergen_list]
            for ingredient in ingredients:
                ingredient.allergens = mapped_allergens

        # Apply learned rules
        ingredients = self.rule_engine.apply_learned_rules(ingredients, language="pt")

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
            id=scalar_fields.get("id") or source_file.split(".")[0],
            name=scalar_fields.get("name", "Unknown"),
            category=scalar_fields.get("category", "Other"),
            country=scalar_fields.get("country"),
            region=scalar_fields.get("region"),
            continent=scalar_fields.get("continent"),
            servings=self._parse_int(scalar_fields.get("servings", 1)),
            prep_time_minutes=None,
            cooking_time_minutes=None,
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
        ing_counter = 0
        step_counter = 0

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

                    # Skip rows with no useful data
                    product_name = str(canonical_row.get("product_name", "")).strip() if "product_name" in canonical_row else ""
                    action = str(canonical_row.get("action", "")).strip() if "action" in canonical_row else ""

                    # Categorize as ingredient or step
                    # Prioritize: if it looks like an action, treat as step (even if in product_name field)
                    if action or self._is_step_action(product_name):
                        # This is a step row
                        step_counter += 1
                        steps.append(self._build_step(canonical_row, step_counter))
                    elif product_name and len(product_name) > 1 and not self._is_step_action(product_name):
                        # Check if this is a compound ingredient (multiple ingredients in one cell)
                        compound_ingredients = self._parse_compound_ingredient(product_name)
                        if compound_ingredients:
                            # Multiple ingredients detected - create separate entries
                            for compound_name, compound_qty, compound_unit in compound_ingredients:
                                ing_counter += 1
                                # Create a modified row for each compound ingredient
                                compound_row = canonical_row.copy()
                                compound_row["product_name"] = compound_name
                                compound_row["quantity"] = compound_qty
                                compound_row["unit"] = compound_unit
                                ingredients.append(self._build_ingredient(compound_row, restaurant_id, ing_counter))
                        else:
                            # Single ingredient row
                            ing_counter += 1
                            ingredients.append(self._build_ingredient(canonical_row, restaurant_id, ing_counter))

        # Apply learned rules to enhance ingredient detection
        ingredients = self.rule_engine.apply_learned_rules(ingredients, language="pt")

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

    def _build_ingredient(self, canonical_row: dict, restaurant_id: str, default_line_number: int = 1) -> Ingredient:
        """Build an Ingredient from normalized row with allergen detection."""
        line_number = self._parse_int(canonical_row.get("line_number"))
        if line_number is None:
            line_number = default_line_number
        product_name = str(canonical_row.get("product_name", "Unknown"))
        quantity = self._parse_float(canonical_row.get("quantity", 0.0))
        unit = self.pattern_store.normalize_unit(
            str(canonical_row.get("unit", "UN")),
            restaurant_id,
        )
        unit_price = self._parse_float(canonical_row.get("unit_price", 0.0))
        observations = canonical_row.get("observations")

        # Allergen detection (MAIN OBJECTIVE)
        allergen_text = f"{product_name} {observations or ''}".strip()
        detected_allergens = self.allergen_db.detect_allergens(allergen_text)
        allergen_list = sorted(list(detected_allergens))

        # Semantic graph enhancement - predict additional allergens from learned relationships
        semantic_predictions = self.rule_engine.correlation_analyzer.predict_allergens(
            product_name, confidence_threshold=0.6
        )
        for pred in semantic_predictions:
            if pred["allergen"] not in allergen_list:
                allergen_list.append(pred["allergen"])
        allergen_list = sorted(list(set(allergen_list)))

        # Risk assessment
        risk_info = assess_allergen_risk(detected_allergens, quantity, unit)
        allergen_risk = risk_info.get("overall_risk", "none")
        allergen_confidence = risk_info.get("confidence", 0.0)

        # Producer & transport extraction
        producer = self.ingredient_analyzer.extract_producer(product_name)
        origin = self.ingredient_analyzer.extract_origin(allergen_text)
        storage_conditions = self.ingredient_analyzer.extract_storage_conditions(observations)

        return Ingredient(
            line_number=line_number,
            product_name=product_name,
            quantity=quantity,
            unit=unit,
            unit_price=unit_price,
            observations=observations,
            allergens=allergen_list,
            allergen_risk=allergen_risk,
            allergen_confidence=allergen_confidence,
            producer=producer,
            origin=origin,
            storage_conditions=storage_conditions,
        )

    def _build_step(self, canonical_row: dict, default_step_number: int = 1) -> PreparationStep:
        """Build a PreparationStep from normalized row."""
        step_number = self._parse_int(canonical_row.get("step_number"))
        if step_number is None:
            step_number = default_step_number
        return PreparationStep(
            step_number=step_number,
            action=str(canonical_row.get("action", "Prepare")),
            product=canonical_row.get("product"),
            cut_type=canonical_row.get("cut_type"),
            time_minutes=self._parse_int(canonical_row.get("time_minutes")),
            observations=canonical_row.get("observations"),
        )

    def _parse_compound_ingredient(self, product_text: str) -> list[tuple[str, float, str]]:
        """
        Parse compound ingredient text (multiple ingredients separated by semicolons).
        Returns: list of (product_name, quantity, unit) tuples, or empty list if single ingredient.

        Examples:
        - "Novilho bife 180g; Batata palito 250g" → [("Novilho bife", 180, "g"), ("Batata palito", 250, "g")]
        - "Ovo 1 un; Sal q.b." → [("Ovo", 1, "un"), ("Sal", 0, "q.b.")]
        - "Single ingredient 100g" → [] (single ingredient, let caller handle normally)
        """
        if not product_text or ";" not in product_text:
            return []

        result = []
        parts = product_text.split(";")

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Parse ingredient: extract name, quantity, unit
            # Pattern: "ProductName qty unit" or "ProductName qty" or "ProductName unit" or just "ProductName"
            # Examples: "Novilho bife 180g", "Ovo 1 un", "Sal q.b.", "Coentros 3g"

            # Try to match: text + optional(number + optional(unit))
            # Use regex: (.*?)\s*([\d.]+)?\s*([a-záéíóúãõç]+\.?|ml|l|un|g|kg)?$
            match = re.match(r'^(.*?)\s+([\d.]+)\s*([a-záéíóúãõçñ]+\.?|ml|l|un|g|kg|oz)?$', part, re.IGNORECASE)

            if match:
                name = match.group(1).strip()
                qty_str = match.group(2)
                unit = match.group(3) or "UN"
                qty = self._parse_float(qty_str)
            else:
                # Try alternate pattern for "name q.b." format
                match = re.match(r'^(.*?)\s+(q\.b\.|qs|quanto\s+baste)$', part, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    qty = 0.0
                    unit = "q.b."
                else:
                    # Just product name, no quantity info
                    name = part
                    qty = 0.0
                    unit = "UN"

            if name and len(name) > 1:
                result.append((name, qty, unit))

        return result

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
    def _is_step_action(text: str) -> bool:
        """Check if text looks like a step action (cooking verb)."""
        if not text:
            return False
        text_lower = text.lower().strip()
        # Common cooking verbs (Portuguese, Spanish, English)
        step_verbs = {
            "temperar", "fritar", "aquecer", "cozer", "assar", "moer", "picar",
            "misturar", "mexer", "deitar", "colocar", "juntar", "cobrir", "decorar",
            "cozinhar", "cozedura", "acção", "accion", "action",
            "fry", "cook", "heat", "boil", "roast", "grind", "chop", "mix",
            "stir", "add", "place", "join", "cover", "decorate",
            "freír", "cocinar", "calentar", "hervir", "asar", "moler", "picar",
        }
        return any(verb in text_lower for verb in step_verbs)

    @staticmethod
    def _parse_float(value) -> float:
        """Safely parse float."""
        try:
            return float(str(value)) if value else 0.0
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _parse_quantity_and_unit(text: str) -> tuple[float, str]:
        """
        Parse quantity and unit from text like "250g", "80 ml", "5 fls".
        Returns (quantity: float, unit: str normalized to uppercase).
        """
        if not text:
            return 0.0, "UN"

        text = str(text).strip()

        # Try to extract number from the beginning
        import re
        match = re.match(r'^([0-9]*\.?[0-9]+)\s*(.*)$', text)

        if match:
            qty_str, unit_str = match.groups()
            try:
                quantity = float(qty_str)
            except ValueError:
                quantity = 0.0

            unit = unit_str.strip().upper() if unit_str else "UN"

            # Normalize common unit abbreviations
            unit_map = {
                "G": "G",
                "GR": "G",
                "GRS": "G",
                "GRAMAS": "G",
                "ML": "ML",
                "L": "L",
                "LT": "L",
                "LITRO": "L",
                "LITROS": "L",
                "KG": "KG",
                "K": "KG",
                "QUILOS": "KG",
                "QUILO": "KG",
                "UN": "UN",
                "UNIDADE": "UN",
                "UNIDADES": "UN",
                "UNIT": "UN",
                "UNITS": "UN",
                "PIECE": "UN",
                "PIECES": "UN",
                "FLS": "UN",  # Folhas/leaves
                "FOLHAS": "UN",
                "FOLHA": "UN",
            }

            # Map abbreviations to canonical units
            normalized_unit = unit_map.get(unit, unit if unit in ["G", "ML", "L", "KG", "UN"] else "UN")

            return quantity, normalized_unit
        else:
            # No number found, return defaults
            return 0.0, "UN"
