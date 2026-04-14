from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Ingredient(BaseModel):
    line_number: int = Field(ge=1)
    product_name: str
    quantity: float = Field(ge=0.0)
    unit: str  # KG, L, UN, G, ML — normalized to uppercase
    unit_price: float = Field(ge=0.0)
    line_cost: float = Field(ge=0.0)  # always computed: quantity * unit_price
    observations: Optional[str] = None

    @model_validator(mode="after")
    def compute_line_cost(self) -> Ingredient:
        # Override whatever the source said; recompute always
        self.line_cost = round(self.quantity * self.unit_price, 4)
        return self


class PreparationStep(BaseModel):
    step_number: int = Field(ge=1)
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

    @model_validator(mode="after")
    def compute_financials(self) -> NormalizedMenuSheet:
        self.total_cost = round(sum(i.line_cost for i in self.ingredients), 4)
        if self.servings > 0:
            self.cost_per_serving = round(self.total_cost / self.servings, 4)
        return self
