"""
Column name classifier using scikit-learn TF-IDF + LogisticRegression.
Identifies column semantics (product_name, quantity, unit, price, etc.).
"""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction import DictVectorizer

from app.ml.feature_extractor import ColumnFeatures, extract_column_features
from app.ml.pattern_store import PatternStore

# Canonical field labels (possible classification targets)
CANONICAL_FIELDS = {
    "product_name",
    "quantity",
    "unit",
    "unit_price",
    "line_cost",
    "line_number",
    "observations",
    "step_number",
    "action",
    "cut_type",
    "time_minutes",
    "name",
    "category",
    "servings",
    "prep_time_minutes",
    "cooking_time_minutes",
    "country",
    "region",
    "continent",
    "IGNORE",
}

# Cold-start synonym dictionary (PT/ES/EN)
SYNONYM_DICT = {
    # Product/ingredient
    "producto": "product_name",
    "produto": "product_name",
    "ingredient": "product_name",
    "ingrediente": "product_name",
    "item": "product_name",
    "articulo": "product_name",
    "product": "product_name",
    # Quantity
    "cantidad": "quantity",
    "qty": "quantity",
    "qtd": "quantity",
    "quantite": "quantity",
    "quantidade": "quantity",
    "cant": "quantity",
    # Unit
    "unidad": "unit",
    "un": "unit",
    "u": "unit",
    "unit": "unit",
    "unidade": "unit",
    "ud": "unit",
    # Price
    "price": "unit_price",
    "precio": "unit_price",
    "preco": "unit_price",
    "unit price": "unit_price",
    "precio unitario": "unit_price",
    "preço unitário": "unit_price",
    "preço": "unit_price",
    # Cost
    "cost": "line_cost",
    "coste": "line_cost",
    "custo": "line_cost",
    "total": "line_cost",
    "amount": "line_cost",
    # Line number
    "line": "line_number",
    "no": "line_number",
    "num": "line_number",
    "numero": "line_number",
    "nº": "line_number",
    # Step
    "paso": "step_number",
    "passo": "step_number",
    "step": "step_number",
    "etapa": "step_number",
    # Action
    "accion": "action",
    "ação": "action",
    "action": "action",
    "operacion": "action",
    "operação": "action",
    # Category
    "categoria": "category",
    "category": "category",
    "classe": "category",
    "class": "category",
    # Servings
    "raciones": "servings",
    "porciones": "servings",
    "porções": "servings",
    "servings": "servings",
    "servir": "servings",
    # Time
    "tiempo": "prep_time_minutes",
    "tempo": "prep_time_minutes",
    "time": "prep_time_minutes",
    "min": "prep_time_minutes",
    # Observations
    "observaciones": "observations",
    "observações": "observations",
    "notes": "observations",
    "notas": "observations",
    "remarks": "observations",
}


class ColumnClassifier:
    """Classify column headers into canonical field types."""

    def __init__(self, pattern_store: PatternStore, models_dir: Path):
        self.pattern_store = pattern_store
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / "column_classifier.joblib"

        # Build scikit-learn pipeline
        self.pipeline = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("tfidf_char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
                            ("tfidf_word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2))),
                            ("meta", DictVectorizer()),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )

        self.is_trained = False
        self._load_model_if_exists()

    def _load_model_if_exists(self):
        """Load pre-trained model if available."""
        if self.model_path.exists():
            try:
                loaded = joblib.load(self.model_path)
                self.pipeline = loaded["pipeline"]
                self.is_trained = True
            except Exception:
                pass

    def classify_headers(self, headers: list[str]) -> dict[str, str]:
        """
        Classify a list of column headers.
        Returns {raw_header: canonical_field}.
        """
        result = {}

        for header in headers:
            canonical = self.classify_single(header)
            result[header] = canonical

        return result

    def classify_single(self, header: str) -> str:
        """
        Classify a single header.
        Returns canonical field name (or "IGNORE" if unrecognized).
        """
        if not header or not header.strip():
            return "IGNORE"

        header_lower = header.strip().lower()

        # Try synonym dictionary first (always available)
        if header_lower in SYNONYM_DICT:
            return SYNONYM_DICT[header_lower]

        # Try substring match in synonyms
        for syn, canonical in SYNONYM_DICT.items():
            if syn in header_lower:
                return canonical

        # Try trained model if available
        if self.is_trained:
            try:
                features = extract_column_features([header])
                feature = features[0]
                meta = {
                    "position": feature.position,
                    "has_currency": float(feature.has_currency),
                    "is_all_caps": float(feature.is_all_caps),
                    "has_number": float(feature.has_number),
                    "length": feature.length,
                }

                # Predict with probabilities
                X_text = [[header_lower]]
                X_meta = [meta]

                # Build feature vectors manually
                vectorizer_char = self.pipeline.named_steps["features"].transformer_list[0][1]
                vectorizer_word = self.pipeline.named_steps["features"].transformer_list[1][1]
                vectorizer_meta = self.pipeline.named_steps["features"].transformer_list[2][1]

                try:
                    vec_char = vectorizer_char.transform(X_text)
                    vec_word = vectorizer_word.transform(X_text)
                    vec_meta = vectorizer_meta.transform(X_meta)

                    # Combine features
                    X_combined = np.hstack([vec_char.toarray(), vec_word.toarray(), vec_meta.toarray()])
                    proba = self.pipeline.named_steps["clf"].predict_proba(X_combined)[0]
                    max_proba = np.max(proba)

                    if max_proba >= 0.5:  # Confidence threshold
                        pred_idx = np.argmax(proba)
                        pred_class = self.pipeline.named_steps["clf"].classes_[pred_idx]
                        return pred_class
                except Exception:
                    pass
            except Exception:
                pass

        return "IGNORE"

    def train(self, X: list[str], y: list[str]):
        """
        Train the classifier on labeled data.
        X: list of raw header names
        y: list of canonical field names
        """
        if len(X) < 3:
            return  # Need minimum examples

        # Extract features
        features_list = extract_column_features(X)
        meta_list = []
        for feature in features_list:
            meta_list.append({
                "position": feature.position,
                "has_currency": float(feature.has_currency),
                "is_all_caps": float(feature.is_all_caps),
                "has_number": float(feature.has_number),
                "length": feature.length,
            })

        # Fit pipeline
        try:
            # First fit the vectorizers on text
            X_lower = [h.strip().lower() for h in X]
            vectorizer_char = self.pipeline.named_steps["features"].transformer_list[0][1]
            vectorizer_word = self.pipeline.named_steps["features"].transformer_list[1][1]
            vectorizer_char.fit(X_lower)
            vectorizer_word.fit(X_lower)

            # Vectorize
            vec_char = vectorizer_char.transform(X_lower)
            vec_word = vectorizer_word.transform(X_lower)
            vectorizer_meta = self.pipeline.named_steps["features"].transformer_list[2][1]
            vec_meta = vectorizer_meta.fit_transform(meta_list)

            # Combine features
            X_combined = np.hstack([vec_char.toarray(), vec_word.toarray(), vec_meta.toarray()])

            # Train classifier
            self.pipeline.named_steps["clf"].fit(X_combined, y)
            self.is_trained = True

            # Save
            self._save_model()
        except Exception as e:
            print(f"Training error: {e}")

    def _save_model(self):
        """Save trained model to disk."""
        try:
            joblib.dump({"pipeline": self.pipeline}, self.model_path)
        except Exception:
            pass
