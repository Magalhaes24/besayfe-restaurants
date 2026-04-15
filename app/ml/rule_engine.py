"""
Self-learning rule engine that extracts and applies generalizable patterns.
Learns from corrections and feedback to improve future predictions.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

from app.ml.pattern_store import PatternStore
from app.ml.semantic_graph import SemanticGraph, CorrelationAnalyzer
from app.ml.tokenizer import TokenVocabulary


class AllergenRule:
    """A learned rule for allergen detection."""

    def __init__(
        self,
        pattern: str,  # Regex pattern
        allergen: str,
        source_ingredient: str,
        confidence: float = 0.8,
        language: str = "pt",
    ):
        self.pattern = pattern
        self.allergen = allergen
        self.source_ingredient = source_ingredient  # Original ingredient that triggered learning
        self.confidence = confidence
        self.language = language
        self.success_count = 0
        self.failure_count = 0
        self.created_at = datetime.utcnow()

    def test(self, text: str) -> bool:
        """Test if pattern matches text."""
        try:
            return bool(re.search(self.pattern, text, re.IGNORECASE))
        except Exception:
            return False

    def update_confidence(self, success: bool):
        """Update confidence based on feedback."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        # Confidence = success / (success + failure), with minimum 0.5
        total = self.success_count + self.failure_count
        if total > 0:
            self.confidence = max(0.5, self.success_count / total)

    def to_dict(self):
        return {
            "pattern": self.pattern,
            "allergen": self.allergen,
            "source_ingredient": self.source_ingredient,
            "confidence": self.confidence,
            "language": self.language,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at.isoformat(),
        }


class CompoundIngredientRule:
    """A learned rule for parsing compound ingredients."""

    def __init__(
        self,
        restaurant_id: str,
        source_format: str,
        pattern_parts: list[str],  # ["Produto", "Qtd", "Unidade", ...]
        delimiter: str = ";",
        confidence: float = 0.8,
    ):
        self.restaurant_id = restaurant_id
        self.source_format = source_format
        self.pattern_parts = pattern_parts  # Expected structure
        self.delimiter = delimiter
        self.confidence = confidence
        self.success_count = 0
        self.failure_count = 0
        self.created_at = datetime.utcnow()

    def matches_structure(self, data_row: dict) -> bool:
        """Check if a data row matches this pattern."""
        product_field = data_row.get("product_name", "")
        # If it contains delimiter and looks like compound ingredient
        return self.delimiter in str(product_field)

    def update_confidence(self, success: bool):
        """Update confidence based on feedback."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        total = self.success_count + self.failure_count
        if total > 0:
            self.confidence = max(0.5, self.success_count / total)

    def to_dict(self):
        return {
            "restaurant_id": self.restaurant_id,
            "source_format": self.source_format,
            "pattern_parts": self.pattern_parts,
            "delimiter": self.delimiter,
            "confidence": self.confidence,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at.isoformat(),
        }


class RuleEngine:
    """Learns rules from corrections and applies them to improve predictions."""

    def __init__(self, pattern_store: PatternStore):
        self.pattern_store = pattern_store
        self.allergen_rules: list[AllergenRule] = []
        self.compound_rules: list[CompoundIngredientRule] = []
        self.ingredient_name_variants: dict[str, str] = {}  # Learned aliases
        self.origin_patterns: dict[str, str] = {}  # Ingredient → origin mapping
        self.semantic_graph = SemanticGraph(pattern_store)  # Knowledge graph with database persistence
        self.correlation_analyzer = CorrelationAnalyzer(self.semantic_graph)
        self.token_vocabulary = TokenVocabulary(pattern_store)  # Persistent vocabulary learning (BPE-like)
        self.recent_tokens: list = []  # Track recently created tokens for visualization
        self.recent_relationships: list = []  # Track recently created/updated relationships
        self._load_stored_rules()

    def _load_stored_rules(self):
        """Load previously learned rules from database."""
        # This will be populated as we train
        pass

    def learn_from_correction(self, original, corrected, feedback_data: dict):
        """
        Extract generalizable rules from user corrections.
        Called after user submits corrections with good/bad feedback.
        """
        # Learn vocabulary and tokens from corrections
        self._learn_vocabulary_from_correction(original, corrected)

        # Learn allergen patterns
        self._learn_allergen_patterns(original, corrected, feedback_data)

        # Learn compound ingredient structures
        self._learn_compound_structures(original, corrected)

        # Learn ingredient name variants
        self._learn_ingredient_variants(original, corrected)

        # Learn origin/producer patterns
        self._learn_origin_patterns(original, corrected)

        # Store rules
        self._persist_rules()

    def _learn_vocabulary_from_correction(self, original, corrected):
        """
        Learn domain-specific vocabulary and token merges from corrections.
        Builds a persistent vocabulary that improves tokenization over time.
        """
        for orig_ing, corr_ing in zip(original.ingredients, corrected.ingredients):
            # Observe ingredient tokens
            self.token_vocabulary.observe_token(
                corr_ing.product_name, token_type="ingredient", language="pt"
            )

            # Observe allergen tokens
            for allergen in corr_ing.allergens:
                self.token_vocabulary.observe_token(
                    allergen, token_type="allergen", language="pt"
                )

            # Learn token pairs (co-occurrences)
            # Ingredient + allergen pairs
            for allergen in corr_ing.allergens:
                self.token_vocabulary.observe_token_pair(
                    corr_ing.product_name,
                    allergen,
                    context="allergen_implication",
                )

            # Ingredient + origin pairs
            if corr_ing.origin:
                self.token_vocabulary.observe_token(
                    corr_ing.origin, token_type="origin", language="pt"
                )
                self.token_vocabulary.observe_token_pair(
                    corr_ing.product_name,
                    corr_ing.origin,
                    context="origin_source",
                )

            # Ingredient + producer pairs
            if corr_ing.producer:
                self.token_vocabulary.observe_token(
                    corr_ing.producer, token_type="producer", language="pt"
                )
                self.token_vocabulary.observe_token_pair(
                    corr_ing.product_name,
                    corr_ing.producer,
                    context="producer_source",
                )

            # Learn merges for multi-word ingredients
            if " " in corr_ing.product_name:
                words = corr_ing.product_name.split()
                # Learn that consecutive words should merge into the full ingredient
                for i in range(len(words) - 1):
                    merged = " ".join(words[: i + 2])
                    if i == 0 or len(words) > 2:
                        self.token_vocabulary.learn_merge(
                            words[i], words[i + 1], merged, confidence=0.9
                        )

    def _learn_allergen_patterns(self, original, corrected, feedback_data: dict):
        """
        Extract allergen detection patterns from corrections.
        If user marked an ingredient allergen detection as "good", reinforce that pattern.
        If marked as "bad", learn to avoid that pattern.
        Uses semantic graph to propagate allergen implications through ingredient families.
        """
        for i, (orig_ing, corr_ing) in enumerate(zip(
            original.ingredients, corrected.ingredients
        )):
            feedback = feedback_data.get(f"ingredient_{i}", {}).get("feedback")
            if not feedback:
                continue

            # Extract allergen info
            if corr_ing.allergens:
                for allergen in corr_ing.allergens:
                    # Create pattern from ingredient name
                    product_name = corr_ing.product_name.lower()

                    # Extract key words (e.g., "Bacalhau" → "bacalh", "Ovo" → "ovo")
                    key_words = [
                        word
                        for word in re.findall(r"\b\w+\b", product_name)
                        if len(word) > 2
                    ]

                    if key_words:
                        # Create rule with word boundary matching
                        pattern = r"\b(" + "|".join(key_words) + r")\b"
                        rule = AllergenRule(
                            pattern=pattern,
                            allergen=allergen,
                            source_ingredient=corr_ing.product_name,
                            language="pt",
                        )

                        # Update confidence based on feedback
                        rule.update_confidence(success=feedback == "good")

                        # Add or update existing rule
                        self._add_or_update_rule(rule)

                        # Learn in semantic graph (if good feedback)
                        if feedback == "good":
                            # Track the ingredient token
                            self._track_token_creation(corr_ing.product_name, "ingredient", "pt")
                            # Track the allergen token
                            self._track_token_creation(allergen, "allergen", "pt")
                            # Track the implication relationship
                            self._track_relationship_creation(
                                corr_ing.product_name, allergen, "implies", 0.85
                            )
                            # Propagate through semantic graph
                            self.semantic_graph.propagate_allergen_implications(
                                corr_ing.product_name, allergen, confidence=0.85
                            )

    def _learn_compound_structures(self, original, corrected):
        """
        Learn compound ingredient parsing from corrections.
        If user had to fix compound ingredients, reinforce that pattern.
        """
        restaurant_id = original.source_file.split("_")[0] if "_" in original.source_file else "default"

        # Check if any ingredients contain semicolons
        for ing in corrected.ingredients:
            if ";" in ing.product_name or ing._is_new:
                rule = CompoundIngredientRule(
                    restaurant_id=restaurant_id,
                    source_format=original.source_format,
                    pattern_parts=["product_name", "quantity", "unit"],
                    delimiter=";",
                    confidence=0.9,
                )
                rule.success_count += 1
                self._add_or_update_compound_rule(rule)

    def _learn_ingredient_variants(self, original, corrected):
        """
        Learn ingredient name variants (aliases).
        If user corrects "Bacalao" → "Bacalhau", learn this mapping.
        """
        for i, (orig_ing, corr_ing) in enumerate(zip(
            original.ingredients, corrected.ingredients
        )):
            if orig_ing.product_name != corr_ing.product_name:
                # User corrected the name
                orig_lower = orig_ing.product_name.lower().strip()
                corr_lower = corr_ing.product_name.lower().strip()

                if orig_lower != corr_lower and len(corr_lower) > 2:
                    # Learn this as a variant
                    self.ingredient_name_variants[orig_lower] = corr_lower

    def _learn_origin_patterns(self, original, corrected):
        """
        Learn origin/producer information from corrections.
        """
        for ing in corrected.ingredients:
            if ing.origin or ing.producer:
                key = ing.product_name.lower().strip()
                if ing.origin:
                    self.origin_patterns[f"{key}_origin"] = ing.origin
                if ing.producer:
                    self.origin_patterns[f"{key}_producer"] = ing.producer

    def apply_learned_rules(self, ingredients: list, language: str = "pt") -> list:
        """
        Apply learned allergen rules to improve predictions.
        Adds allergens that rules detect with high confidence.
        """
        for ing in ingredients:
            if not ing.allergens:  # Only enhance if no allergens detected yet
                for rule in self.allergen_rules:
                    if (
                        rule.language == language
                        and rule.confidence > 0.7
                        and rule.test(ing.product_name)
                    ):
                        if rule.allergen not in ing.allergens:
                            ing.allergens.append(rule.allergen)

            # Apply ingredient name corrections
            product_lower = ing.product_name.lower().strip()
            if product_lower in self.ingredient_name_variants:
                corrected_name = self.ingredient_name_variants[product_lower]
                # Keep original but note the variant
                ing.observations = (
                    f"Variant of: {corrected_name}. {ing.observations or ''}"
                ).strip()

        return ingredients

    def get_learning_confidence(self) -> dict:
        """Get system confidence metrics."""
        if not self.allergen_rules:
            return {"allergen_rules": 0, "avg_confidence": 0.0, "status": "untrained"}

        avg_conf = sum(r.confidence for r in self.allergen_rules) / len(
            self.allergen_rules
        )
        return {
            "allergen_rules": len(self.allergen_rules),
            "compound_rules": len(self.compound_rules),
            "variants_learned": len(self.ingredient_name_variants),
            "avg_confidence": avg_conf,
            "status": "learning" if len(self.allergen_rules) < 10 else "trained",
        }

    def _add_or_update_rule(self, new_rule: AllergenRule):
        """Add rule or update if similar rule exists."""
        for existing in self.allergen_rules:
            # Check if rule is similar (same allergen and similar pattern)
            if existing.allergen == new_rule.allergen:
                if self._patterns_are_similar(existing.pattern, new_rule.pattern):
                    # Merge confidences
                    existing.update_confidence(True)
                    return

        # Add as new rule
        self.allergen_rules.append(new_rule)

    def _add_or_update_compound_rule(self, new_rule: CompoundIngredientRule):
        """Add or update compound ingredient rule."""
        for existing in self.compound_rules:
            if (
                existing.restaurant_id == new_rule.restaurant_id
                and existing.source_format == new_rule.source_format
            ):
                # Update existing
                existing.success_count += 1
                existing.update_confidence(True)
                return

        # Add as new
        self.compound_rules.append(new_rule)

    @staticmethod
    def _patterns_are_similar(pattern1: str, pattern2: str) -> bool:
        """Check if two regex patterns are similar."""
        # Simple heuristic: compare pattern length and key words
        if abs(len(pattern1) - len(pattern2)) > 10:
            return False
        # Extract words
        words1 = set(re.findall(r"\w+", pattern1))
        words2 = set(re.findall(r"\w+", pattern2))
        overlap = len(words1 & words2) / max(len(words1 | words2), 1)
        return overlap > 0.6

    def _persist_rules(self):
        """Save learned rules to database."""
        # Store in pattern_store as JSON for now
        rules_data = {
            "allergen_rules": [r.to_dict() for r in self.allergen_rules],
            "compound_rules": [r.to_dict() for r in self.compound_rules],
            "variants": self.ingredient_name_variants,
            "origins": self.origin_patterns,
            "created_at": datetime.utcnow().isoformat(),
        }

        # TODO: Store in database instead of memory
        pass

    def get_rules_summary(self) -> dict:
        """Get summary of learned rules for UI."""
        return {
            "total_allergen_rules": len(self.allergen_rules),
            "total_compound_rules": len(self.compound_rules),
            "learned_variants": len(self.ingredient_name_variants),
            "allergen_patterns": [
                {
                    "allergen": r.allergen,
                    "source": r.source_ingredient,
                    "confidence": round(r.confidence, 2),
                    "success_rate": f"{r.success_count}/{r.success_count + r.failure_count}",
                }
                for r in sorted(
                    self.allergen_rules,
                    key=lambda x: x.confidence,
                    reverse=True,
                )[:10]
            ],
            "compound_patterns": [
                {
                    "restaurant": r.restaurant_id,
                    "format": r.source_format,
                    "confidence": round(r.confidence, 2),
                }
                for r in self.compound_rules
            ],
        }

    def get_ingredient_semantic_profile(self, ingredient: str) -> dict:
        """Get semantic analysis of an ingredient including relationships and predictions."""
        profile = self.semantic_graph.get_ingredient_profile(ingredient)

        # Add semantic predictions
        predictions = self.correlation_analyzer.predict_allergens(ingredient)

        return {
            "ingredient": ingredient,
            "profile": profile,
            "allergen_predictions": predictions,
            "semantic_relationships": {
                "families": self.correlation_analyzer.find_ingredient_families(),
                "correlations": self.correlation_analyzer.find_allergen_correlations(),
            },
            "graph_stats": self.semantic_graph.get_graph_statistics(),
        }

    def analyze_semantic_correlations(self) -> dict:
        """Get comprehensive semantic analysis of all learned relationships."""
        return self.correlation_analyzer.analyze_correlations()

    def get_tokenization_activity(self) -> dict:
        """Get recent tokenization and semantic graph activity."""
        return {
            "recent_tokens": self.recent_tokens[-20:],  # Last 20
            "recent_relationships": self.recent_relationships[-20:],  # Last 20
            "graph_snapshot": {
                "total_tokens": len(self.semantic_graph.tokens),
                "total_relationships": len(self.semantic_graph.relationships),
                "top_tokens": [
                    {
                        "value": token.value,
                        "type": token.token_type,
                        "confidence": token.confidence,
                    }
                    for token in list(self.semantic_graph.tokens.values())[:10]
                ],
                "strongest_relationships": [
                    {
                        "source": rel.source.value,
                        "target": rel.target.value,
                        "type": rel.relation_type,
                        "strength": round(rel.strength, 2),
                        "evidence": rel.evidence_count,
                    }
                    for rel in sorted(
                        self.semantic_graph.relationships,
                        key=lambda r: r.strength,
                        reverse=True,
                    )[:10]
                ],
            },
        }

    def learn_from_normalized_sheet(self, sheet):
        """
        Learn from a normalized sheet without waiting for corrections.
        Feeds all detected ingredients and allergens into semantic graph and vocabulary.
        Called immediately after normalization to bootstrap learning from technical sheets.
        """
        for ing in sheet.ingredients:
            # Feed ingredient token to vocabulary
            self.token_vocabulary.observe_token(
                ing.product_name, token_type="ingredient", language="pt"
            )
            self._track_token_creation(ing.product_name, "ingredient", "pt")

            # Feed allergen tokens to vocabulary and semantic graph
            for allergen in ing.allergens:
                self.token_vocabulary.observe_token(
                    allergen, token_type="allergen", language="pt"
                )
                self._track_token_creation(allergen, "allergen", "pt")

                # Learn ingredient-allergen pairing
                self.token_vocabulary.observe_token_pair(
                    ing.product_name,
                    allergen,
                    context="allergen_implication",
                )

                # Add to semantic graph with high confidence (verified data)
                self.semantic_graph.propagate_allergen_implications(
                    ing.product_name, allergen, confidence=0.95
                )
                self._track_relationship_creation(
                    ing.product_name, allergen, "implies", 0.95
                )

            # Feed origin and producer tokens if present
            if ing.origin:
                self.token_vocabulary.observe_token(
                    ing.origin, token_type="origin", language="pt"
                )
                self.token_vocabulary.observe_token_pair(
                    ing.product_name,
                    ing.origin,
                    context="origin_source",
                )
                self._track_token_creation(ing.origin, "origin", "pt")

            if ing.producer:
                self.token_vocabulary.observe_token(
                    ing.producer, token_type="producer", language="pt"
                )
                self.token_vocabulary.observe_token_pair(
                    ing.product_name,
                    ing.producer,
                    context="producer_source",
                )
                self._track_token_creation(ing.producer, "producer", "pt")

            # Learn merges for multi-word ingredients
            if " " in ing.product_name:
                words = ing.product_name.split()
                for i in range(len(words) - 1):
                    merged = " ".join(words[: i + 2])
                    if i == 0 or len(words) > 2:
                        self.token_vocabulary.learn_merge(
                            words[i], words[i + 1], merged, confidence=0.95
                        )

    def _track_token_creation(self, token_value: str, token_type: str, language: str = "pt"):
        """Track token creation for visualization."""
        self.recent_tokens.append({
            "value": token_value,
            "type": token_type,
            "language": language,
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Keep only recent entries
        if len(self.recent_tokens) > 100:
            self.recent_tokens = self.recent_tokens[-100:]

    def _track_relationship_creation(self, source_val: str, target_val: str, rel_type: str, strength: float):
        """Track relationship creation for visualization."""
        self.recent_relationships.append({
            "source": source_val,
            "target": target_val,
            "type": rel_type,
            "strength": round(strength, 2),
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Keep only recent entries
        if len(self.recent_relationships) > 100:
            self.recent_relationships = self.recent_relationships[-100:]

    def get_vocabulary_stats(self) -> dict:
        """Get comprehensive vocabulary learning statistics."""
        vocab_stats = self.token_vocabulary.get_vocabulary_stats()
        merge_candidates = self.token_vocabulary.get_merge_candidates(min_frequency=2)

        return {
            "vocabulary": vocab_stats,
            "merge_candidates": [
                {
                    "token_a": a,
                    "token_b": b,
                    "frequency": freq,
                    "recommendation": "high priority" if freq > 10 else "medium" if freq > 5 else "low",
                }
                for a, b, freq in merge_candidates[:10]
            ],
            "tokenization_quality": {
                "total_tokens_seen": vocab_stats["total_frequency"],
                "unique_tokens": vocab_stats["total_tokens"],
                "learned_merges": vocab_stats["total_merges"],
                "vocabulary_coverage": round(
                    (vocab_stats["total_tokens"] / max(1, vocab_stats["total_frequency"])) * 100, 1
                ),
            },
        }
