from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Ingredient(BaseModel):
    line_number: int = Field(default=1, ge=1)
    product_name: str
    quantity: float = Field(ge=0.0)
    unit: str  # KG, L, UN, G, ML — normalized to uppercase
    unit_price: float = Field(ge=0.0)
    line_cost: float = Field(default=0.0, ge=0.0)  # always computed: quantity * unit_price
    observations: Optional[str] = None

    # Allergen tracking (main objective)
    allergens: list[str] = Field(default_factory=list)  # e.g., ["milk", "eggs", "shellfish"]
    allergen_risk: str = Field(default="none")  # "critical", "high", "medium", "low", "none"
    allergen_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Producer & transport tracking
    producer: Optional[str] = None  # Brand/manufacturer name
    origin: Optional[str] = None  # Country/region of origin
    storage_conditions: Optional[str] = None  # Temperature, humidity, transport conditions

    @model_validator(mode="after")
    def compute_line_cost(self) -> Ingredient:
        # Override whatever the source said; recompute always
        self.line_cost = round(self.quantity * self.unit_price, 4)
        return self


class PreparationStep(BaseModel):
    step_number: int = Field(default=1, ge=1)
    action: str  # Temperar, Fritar, Misturar, etc.
    product: Optional[str] = None
    cut_type: Optional[str] = None
    time_minutes: Optional[int] = Field(default=None, ge=0)
    observations: Optional[str] = None


class NormalizedMenuSheet(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Recipe identity
    name: str
    category: str  # Carne, Peixe, Vegetariano, etc.
    country: Optional[str] = None
    region: Optional[str] = None
    continent: Optional[str] = None

    # Times
    prep_time_minutes: Optional[int] = Field(default=None, ge=0)
    cooking_time_minutes: Optional[int] = Field(default=None, ge=0)

    # Servings
    servings: int = Field(ge=1)

    # Ingredients and steps
    ingredients: list[Ingredient]
    steps: list[PreparationStep]

    # Computed financials — always derived, never trusted from source
    total_cost: float = Field(default=0.0)
    cost_per_serving: float = Field(default=0.0)

    # Metadata
    source_file: str
    source_format: str  # pdf | csv | xlsx
    normalized_at: datetime = Field(default_factory=datetime.utcnow)
    confidence_score: float = Field(ge=0.0, le=1.0)

    # Raw extraction for auditability
    raw_extraction: dict = Field(default_factory=dict)

    # Allergen summary (MAIN OBJECTIVE)
    all_allergens: list[str] = Field(default_factory=list)  # All unique allergens in dish
    critical_allergens: list[str] = Field(default_factory=list)  # Critical allergens detected
    allergen_ingredients_count: int = Field(default=0)  # Number of ingredients with allergens
    has_critical_allergen: bool = Field(default=False)  # True if any critical allergens detected
    allergen_risk_level: str = Field(default="none")  # Overall risk: critical, high, medium, low, none

    @model_validator(mode="after")
    def compute_financials_and_allergens(self) -> NormalizedMenuSheet:
        # Compute financials
        self.total_cost = round(sum(i.line_cost for i in self.ingredients), 4)
        if self.servings > 0:
            self.cost_per_serving = round(self.total_cost / self.servings, 4)

        # Compute allergen summary
        all_allergen_set = set()
        critical_allergen_set = set()
        allergen_count = 0

        for ing in self.ingredients:
            if ing.allergens:
                allergen_count += 1
                for allergen in ing.allergens:
                    all_allergen_set.add(allergen)
                    # Critical allergens: milk, eggs, peanuts, tree_nuts, fish, shellfish, soy, wheat
                    if allergen in {"milk", "eggs", "peanuts", "tree_nuts", "fish", "shellfish", "soy", "wheat"}:
                        critical_allergen_set.add(allergen)

        self.all_allergens = sorted(list(all_allergen_set))
        self.critical_allergens = sorted(list(critical_allergen_set))
        self.allergen_ingredients_count = allergen_count
        self.has_critical_allergen = len(critical_allergen_set) > 0

        # Determine overall risk level
        if self.has_critical_allergen:
            self.allergen_risk_level = "critical"
        elif all_allergen_set:
            self.allergen_risk_level = "high"
        else:
            self.allergen_risk_level = "none"

        return self
