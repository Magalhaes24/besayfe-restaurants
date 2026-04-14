"""
Trainer for retraining classifiers after user corrections.
Extracts learning signals from correction diffs and updates models.
"""

from __future__ import annotations

import json
from typing import Optional

from app.ml.column_classifier import ColumnClassifier
from app.ml.pattern_store import PatternStore


class Trainer:
    """Retrains ML models based on user corrections."""

    def __init__(self, pattern_store: PatternStore):
        self.pattern_store = pattern_store

    async def process_correction(
        self,
        correction_id: int,
        diff_json: str,
        source_fmt: str,
        restaurant_id: Optional[str] = None,
    ):
        """
        Process a correction diff and extract learning signals.
        Retrains classifier if thresholds are met.
        """
        try:
            diff = json.loads(diff_json)
        except json.JSONDecodeError:
            return

        # Extract learning signals
        for field_path, change in diff.items():
            if not isinstance(change, dict) or "corrected" not in change:
                continue

            predicted = change.get("predicted")
            corrected = change.get("corrected")

            # Learn unit normalization
            if "unit" in field_path:
                if predicted and corrected:
                    self.pattern_store.add_unit_mapping(
                        raw_unit=str(predicted),
                        canonical=str(corrected),
                        restaurant=restaurant_id,
                        origin="correction",
                    )

            # Learn column mappings (for CSV/XLSX)
            if source_fmt in ("csv", "xlsx") and field_path.startswith("column_"):
                raw_name = field_path.replace("column_", "")
                if corrected:
                    self.pattern_store.add_column_mapping(
                        raw_name=raw_name,
                        canonical=str(corrected),
                        source_fmt=source_fmt,
                        restaurant=restaurant_id,
                        origin="correction",
                    )

        # Check if we have enough examples to retrain
        self._maybe_retrain_classifier(source_fmt)

        # Mark as applied
        self.pattern_store.mark_correction_applied(correction_id)

    def _maybe_retrain_classifier(self, source_fmt: str):
        """Retrain column classifier if enough labeled examples exist."""
        if source_fmt not in ("csv", "xlsx"):
            return

        # Get all column examples
        all_examples = self.pattern_store.get_all_column_examples("")

        if len(all_examples) < 5:
            return  # Not enough examples

        # TODO: Collect training data and retrain
        # This would involve querying all column_mappings and training the classifier
