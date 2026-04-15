"""
Semantic relationship graph for understanding ingredient correlations.
Creates a knowledge graph of relationships between ingredients, allergens, origins, etc.
Uses tokenization to identify semantic connections and improve prediction accuracy.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

from app.ml.allergen_detector import AllergenDatabase


class Token:
    """Represents a semantic unit (ingredient, allergen, origin, etc.)."""

    def __init__(
        self,
        value: str,
        token_type: str,  # "ingredient", "allergen", "origin", "producer", "preparation"
        language: str = "pt",
        confidence: float = 1.0,
    ):
        self.value = value.lower().strip()
        self.token_type = token_type
        self.language = language
        self.confidence = confidence
        self.created_at = datetime.utcnow()

    def __hash__(self):
        return hash((self.value, self.token_type))

    def __eq__(self, other):
        if not isinstance(other, Token):
            return False
        return self.value == other.value and self.token_type == other.token_type

    def __repr__(self):
        return f"Token({self.token_type}:{self.value}@{self.confidence:.2f})"

    def to_dict(self):
        return {
            "value": self.value,
            "type": self.token_type,
            "language": self.language,
            "confidence": self.confidence,
        }


class Relationship:
    """Represents a relationship between two tokens."""

    def __init__(
        self,
        source: Token,
        target: Token,
        relation_type: str,  # "implies", "variant_of", "contains", "produced_by", "from"
        strength: float = 0.5,  # 0-1, confidence of relationship
        evidence_count: int = 1,
    ):
        self.source = source
        self.target = target
        self.relation_type = relation_type
        self.strength = strength  # Updated based on evidence
        self.evidence_count = evidence_count
        self.created_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()

    def reinforce(self):
        """Strengthen this relationship based on additional evidence."""
        self.evidence_count += 1
        # Strength increases with evidence, but caps at 0.99
        self.strength = min(0.99, 0.5 + (self.evidence_count / 20))
        self.last_updated = datetime.utcnow()

    def weaken(self):
        """Weaken this relationship (conflicting evidence)."""
        self.strength = max(0.1, self.strength - 0.1)
        self.last_updated = datetime.utcnow()

    def to_dict(self):
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "type": self.relation_type,
            "strength": round(self.strength, 2),
            "evidence_count": self.evidence_count,
        }

    def __repr__(self):
        return f"{self.source.value} --[{self.relation_type}@{self.strength:.2f}]--> {self.target.value}"


class SemanticGraph:
    """Knowledge graph of semantic relationships."""

    def __init__(self, pattern_store=None):
        self.tokens: dict[tuple, Token] = {}  # (value, type) -> Token
        self.relationships: list[Relationship] = []
        self.adjacency: dict[Token, list[Relationship]] = defaultdict(list)
        self.allergen_db = AllergenDatabase()
        self.pattern_store = pattern_store

        # Load from database if available
        if pattern_store:
            self._load_from_db()

    def _load_from_db(self):
        """Load semantic graph from database."""
        if not self.pattern_store:
            return

        try:
            data = self.pattern_store.load_semantic_graph()

            # Reconstruct tokens
            for key, token_data in data.get("tokens", {}).items():
                token = Token(
                    text=token_data["value"],
                    token_type=token_data["type"],
                    language=token_data.get("language", "pt"),
                    confidence=token_data.get("confidence", 1.0),
                )
                if token_data.get("created_at"):
                    token.created_at = datetime.fromisoformat(token_data["created_at"])
                if token_data.get("last_used"):
                    token.last_used = datetime.fromisoformat(token_data["last_used"])
                self.tokens[key] = token

            # Reconstruct relationships
            for rel_data in data.get("relationships", []):
                source = self.add_token(
                    rel_data["source_value"],
                    rel_data["source_type"]
                )
                target = self.add_token(
                    rel_data["target_value"],
                    rel_data["target_type"]
                )

                rel = Relationship(
                    source=source,
                    target=target,
                    relation_type=rel_data["relation_type"],
                    strength=rel_data.get("strength", 0.5),
                    evidence_count=rel_data.get("evidence_count", 1),
                )
                if rel_data.get("created_at"):
                    rel.created_at = datetime.fromisoformat(rel_data["created_at"])
                self.relationships.append(rel)
                self.adjacency[source].append(rel)
        except Exception as e:
            print(f"Warning: Failed to load semantic graph from database: {e}")

    def add_token(
        self, value: str, token_type: str, language: str = "pt", confidence: float = 1.0
    ) -> Token:
        """Add or get a token."""
        key = (value.lower().strip(), token_type)
        if key not in self.tokens:
            self.tokens[key] = Token(value, token_type, language, confidence)
            # Persist to database
            if self.pattern_store:
                self.pattern_store.save_semantic_token(value, token_type, language, confidence)
        return self.tokens[key]

    def add_relationship(
        self,
        source_val: str,
        source_type: str,
        target_val: str,
        target_type: str,
        relation_type: str,
        strength: float = 0.5,
    ) -> Relationship:
        """Add a relationship between tokens."""
        source = self.add_token(source_val, source_type)
        target = self.add_token(target_val, target_type)

        # Check if relationship already exists
        for rel in self.adjacency[source]:
            if rel.target == target and rel.relation_type == relation_type:
                rel.reinforce()
                # Persist updated relationship
                if self.pattern_store:
                    self.pattern_store.save_semantic_relationship(
                        source_val, source_type, target_val, target_type, relation_type, rel.strength
                    )
                return rel

        # Create new relationship
        rel = Relationship(source, target, relation_type, strength)
        self.relationships.append(rel)
        self.adjacency[source].append(rel)

        # Persist to database
        if self.pattern_store:
            self.pattern_store.save_semantic_relationship(
                source_val, source_type, target_val, target_type, relation_type, strength
            )

        return rel

    def extract_tokens_from_ingredient(self, ingredient_text: str) -> list[Token]:
        """Extract semantic tokens from ingredient text."""
        tokens = []
        text_lower = ingredient_text.lower()

        # Extract ingredient name (main token)
        ing_token = self.add_token(ingredient_text, "ingredient", language="pt")
        tokens.append(ing_token)

        # Extract allergens
        allergens = self.allergen_db.detect_allergens(ingredient_text)
        for allergen in allergens:
            allergen_token = self.add_token(allergen, "allergen")
            tokens.append(allergen_token)
            # Create implication relationship: ingredient implies allergen
            self.add_relationship(
                ingredient_text,
                "ingredient",
                allergen,
                "allergen",
                "implies",
                strength=0.9,
            )

        # Extract common preparation methods
        prep_verbs = {
            "fresco": "fresh",
            "cru": "raw",
            "cozido": "cooked",
            "assado": "roasted",
            "frito": "fried",
            "grelhado": "grilled",
            "fumado": "smoked",
        }
        for keyword, prep_type in prep_verbs.items():
            if keyword in text_lower:
                prep_token = self.add_token(prep_type, "preparation")
                tokens.append(prep_token)
                self.add_relationship(
                    ingredient_text, "ingredient", prep_type, "preparation", "prepared_as"
                )

        # Extract origin/region hints
        origins = {
            "portugal": "Portugal",
            "português": "Portugal",
            "brasil": "Brazil",
            "brasileiro": "Brazil",
            "españa": "Spain",
            "espanha": "Spain",
            "españa": "Spain",
            "europeu": "Europe",
        }
        for keyword, origin in origins.items():
            if keyword in text_lower:
                origin_token = self.add_token(origin, "origin")
                tokens.append(origin_token)
                self.add_relationship(
                    ingredient_text, "ingredient", origin, "origin", "from"
                )

        return tokens

    def find_related(
        self, token: Token, relation_types: list[str] = None, depth: int = 1
    ) -> list[Relationship]:
        """Find related tokens (graph traversal)."""
        results = []

        # Direct relationships
        for rel in self.adjacency.get(token, []):
            if relation_types is None or rel.relation_type in relation_types:
                results.append(rel)

        # Deeper relationships (if depth > 1)
        if depth > 1:
            for rel in results.copy():
                deeper = self.find_related(rel.target, relation_types, depth - 1)
                results.extend(deeper)

        # Remove duplicates
        seen = set()
        unique = []
        for rel in results:
            key = (rel.source.value, rel.target.value, rel.relation_type)
            if key not in seen:
                seen.add(key)
                unique.append(rel)

        return unique

    def find_similar_ingredients(
        self, ingredient: str, threshold: float = 0.6
    ) -> list[Token]:
        """Find similar ingredients using relationship graph."""
        query_token = self.add_token(ingredient, "ingredient")
        similar = []

        # Find ingredients that share allergens
        allergen_rels = self.find_related(
            query_token, relation_types=["implies"], depth=1
        )
        allergens = {rel.target for rel in allergen_rels}

        # Find other ingredients with same allergens
        for token, rels in self.adjacency.items():
            if token.token_type == "ingredient" and token != query_token:
                token_allergens = {
                    rel.target for rel in rels if rel.relation_type == "implies"
                }
                # Similarity = intersection / union of allergens
                if allergens and token_allergens:
                    similarity = len(allergens & token_allergens) / len(
                        allergens | token_allergens
                    )
                    if similarity >= threshold:
                        similar.append((token, similarity))

        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        return [token for token, _ in similar]

    def propagate_allergen_implications(
        self, ingredient: str, allergen: str, confidence: float = 0.85
    ):
        """
        Learn and propagate allergen implications through the graph.
        If ingredient X has allergen Y, related ingredients likely do too.
        """
        ing_token = self.add_token(ingredient, "ingredient")
        allergen_token = self.add_token(allergen, "allergen")

        # Add direct relationship
        self.add_relationship(ingredient, "ingredient", allergen, "allergen", "implies", confidence)

        # Propagate to similar ingredients (with reduced confidence)
        similar = self.find_similar_ingredients(ingredient, threshold=0.5)
        for sim_token in similar:
            # Only propagate if not already related
            existing = any(
                rel.target == allergen_token
                for rel in self.adjacency[sim_token]
                if rel.relation_type == "implies"
            )
            if not existing:
                # Propagated relationships have lower confidence
                self.add_relationship(
                    sim_token.value,
                    "ingredient",
                    allergen,
                    "allergen",
                    "implies",
                    confidence * 0.7,
                )

    def get_ingredient_profile(self, ingredient: str) -> dict:
        """Get complete profile of an ingredient."""
        token = self.add_token(ingredient, "ingredient")
        profile = {
            "ingredient": ingredient,
            "allergens": [],
            "origins": [],
            "preparations": [],
            "related_ingredients": [],
            "confidence": {},
        }

        for rel in self.adjacency.get(token, []):
            if rel.relation_type == "implies":
                profile["allergens"].append(rel.target.value)
                profile["confidence"][rel.target.value] = rel.strength
            elif rel.relation_type == "from":
                profile["origins"].append(rel.target.value)
            elif rel.relation_type == "prepared_as":
                profile["preparations"].append(rel.target.value)

        # Add similar ingredients
        similar = self.find_similar_ingredients(ingredient)
        profile["related_ingredients"] = [t.value for t in similar[:5]]

        return profile

    def get_graph_statistics(self) -> dict:
        """Get statistics about the graph."""
        ingredient_tokens = [t for t in self.tokens.values() if t.token_type == "ingredient"]
        allergen_tokens = [t for t in self.tokens.values() if t.token_type == "allergen"]

        implication_rels = [r for r in self.relationships if r.relation_type == "implies"]
        variant_rels = [r for r in self.relationships if r.relation_type == "variant_of"]

        return {
            "total_tokens": len(self.tokens),
            "ingredients": len(ingredient_tokens),
            "allergens": len(allergen_tokens),
            "total_relationships": len(self.relationships),
            "implications": len(implication_rels),
            "variants": len(variant_rels),
            "avg_relationship_strength": sum(r.strength for r in self.relationships) / max(1, len(self.relationships)),
            "created_at": datetime.utcnow().isoformat(),
        }

    def to_dict(self) -> dict:
        """Export graph as dictionary."""
        return {
            "tokens": {str(k): v.to_dict() for k, v in self.tokens.items()},
            "relationships": [r.to_dict() for r in self.relationships],
            "statistics": self.get_graph_statistics(),
        }


class CorrelationAnalyzer:
    """Analyzes correlations in the semantic graph."""

    def __init__(self, graph: SemanticGraph):
        self.graph = graph

    def find_allergen_correlations(self, min_confidence: float = 0.6) -> dict:
        """Find which ingredients frequently appear together with allergens."""
        correlations = defaultdict(list)

        for rel in self.graph.relationships:
            if rel.relation_type == "implies" and rel.strength >= min_confidence:
                correlations[rel.target.value].append(
                    {
                        "ingredient": rel.source.value,
                        "strength": rel.strength,
                        "evidence": rel.evidence_count,
                    }
                )

        return dict(correlations)

    def find_ingredient_families(self) -> dict:
        """Identify ingredient families based on shared allergens."""
        families = defaultdict(set)

        # Group ingredients by their allergen profile
        for token in self.graph.tokens.values():
            if token.token_type == "ingredient":
                allergens = tuple(
                    sorted(
                        [
                            rel.target.value
                            for rel in self.graph.adjacency.get(token, [])
                            if rel.relation_type == "implies"
                        ]
                    )
                )
                if allergens:
                    families[allergens].add(token.value)

        result = {}
        for allergens, ingredients in families.items():
            family_key = "family_" + "_".join(allergens)
            ingredient_list = list(ingredients)
            result[family_key] = ingredient_list

            # Persist family to database
            if self.graph.pattern_store:
                import json
                try:
                    self.graph.pattern_store.update_ingredient_family(
                        family_key=family_key,
                        allergen_profile=",".join(allergens),
                        ingredients_json=json.dumps(ingredient_list),
                        member_count=len(ingredient_list),
                    )
                except Exception as e:
                    pass  # Non-critical

        return result

    def predict_allergens(self, ingredient: str, confidence_threshold: float = 0.6) -> list[str]:
        """Predict allergens for an ingredient using graph knowledge."""
        token = self.graph.add_token(ingredient, "ingredient")
        predictions = []

        for rel in self.graph.adjacency.get(token, []):
            if (
                rel.relation_type == "implies"
                and rel.strength >= confidence_threshold
            ):
                predictions.append(
                    {
                        "allergen": rel.target.value,
                        "confidence": rel.strength,
                        "source": "learned",
                    }
                )

        return predictions

    def analyze_correlations(self) -> dict:
        """Full correlation analysis of the graph."""
        return {
            "allergen_correlations": self.find_allergen_correlations(),
            "ingredient_families": self.find_ingredient_families(),
            "graph_statistics": self.graph.get_graph_statistics(),
        }
