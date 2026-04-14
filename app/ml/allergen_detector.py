"""
Allergen detection and ingredient analysis module.
Identifies allergens, producers, and transport conditions from ingredient data.
Supports PT, ES, EN languages.
"""

from __future__ import annotations

import re
from typing import Optional, Set

# ────────────────────────────────────────────────────────────────────────────
# ALLERGEN DATABASE
# ────────────────────────────────────────────────────────────────────────────

class AllergenDatabase:
    """Database of common allergens and their indicators in multiple languages."""

    # Top allergens (14 EU + common ones)
    ALLERGEN_GROUPS = {
        "milk": {
            "en": ["milk", "dairy", "butter", "cheese", "cream", "yogurt", "lactose", "whey", "casein", "ghee"],
            "pt": ["leite", "laticínio", "manteiga", "queijo", "creme", "iogurte", "lactose", "soro", "caseína"],
            "es": ["leche", "lácteo", "mantequilla", "queso", "crema", "yogur", "lactosa", "suero", "caseína"],
        },
        "eggs": {
            "en": ["egg", "eggs", "albumin", "mayonnaise", "lecithin"],
            "pt": ["ovo", "ovos", "albumina", "maionese", "lecitina"],
            "es": ["huevo", "huevos", "albúmina", "mayonesa", "lecitina"],
        },
        "peanuts": {
            "en": ["peanut", "peanuts", "groundnut", "arachis", "arachide"],
            "pt": ["amendoim", "amendoins", "amendoim-do-chão", "arachis"],
            "es": ["cacahuete", "cacahuetes", "maní", "arachis"],
        },
        "tree_nuts": {
            "en": ["almond", "hazelnut", "walnut", "cashew", "pistachio", "macadamia", "pecan", "brazil nut", "chestnut", "pine nut"],
            "pt": ["amêndoa", "avelã", "noz", "castanha de caju", "pistáchio", "macadâmia", "noz-pecan", "castanha do pará", "castanha", "pinhão"],
            "es": ["almendra", "avellana", "nuez", "anacardo", "pistacho", "macadamia", "nuez pecan", "castaña de brasil", "castaña", "piñón"],
        },
        "fish": {
            "en": ["fish", "salmon", "tuna", "cod", "trout", "anchovy", "herring"],
            "pt": ["peixe", "salmão", "atum", "bacalau", "truta", "anchova", "arenque"],
            "es": ["pez", "salmón", "atún", "bacalao", "trucha", "anchoa", "arenque"],
        },
        "shellfish": {
            "en": ["shellfish", "shrimp", "prawn", "crab", "lobster", "oyster", "mussel", "clam", "squid", "scallop"],
            "pt": ["marisco", "camarão", "camarao", "gambas", "caranguejo", "lagosta", "ostra", "mexilhão", "mexilhao", "amêijoa", "ameijoa", "lula", "vieira"],
            "es": ["marisco", "camarón", "camaron", "gamba", "cangrejo", "langosta", "ostra", "mejillón", "mejillon", "almeja", "calamar", "vieira"],
        },
        "soy": {
            "en": ["soy", "soybean", "soya", "tofu", "edamame", "tamari", "tempeh", "soy sauce"],
            "pt": ["soja", "soja", "tofu", "edamame", "tamari", "tempeh", "molho de soja"],
            "es": ["soja", "soja", "tofu", "edamame", "tamari", "tempeh", "salsa de soja"],
        },
        "wheat": {
            "en": ["wheat", "gluten", "spelt", "kamut", "farina", "semolina", "bulgur"],
            "pt": ["trigo", "glúten", "espelta", "kamut", "farinha", "sêmola", "bulgur"],
            "es": ["trigo", "gluten", "espelta", "kamut", "harina", "sémola", "bulgur"],
        },
        "barley": {
            "en": ["barley", "malt", "pearl barley"],
            "pt": ["cevada", "malte", "cevada perlada"],
            "es": ["cebada", "malta", "cebada perlada"],
        },
        "rye": {
            "en": ["rye", "rye flour"],
            "pt": ["centeio", "farinha de centeio"],
            "es": ["centeno", "harina de centeno"],
        },
        "oats": {
            "en": ["oat", "oats", "oatmeal"],
            "pt": ["aveia", "aveia", "flocos de aveia"],
            "es": ["avena", "avena", "copos de avena"],
        },
        "sesame": {
            "en": ["sesame", "tahini", "halva", "hummus"],
            "pt": ["gergelim", "tahine", "halvá", "hummus"],
            "es": ["sésamo", "tahini", "halva", "hummus"],
        },
        "sulphites": {
            "en": ["sulphite", "sulfite", "sulphur dioxide", "sulfur dioxide", "SO2", "wine", "vinho", "vino"],
            "pt": ["sulfito", "dióxido de enxofre", "vinho", "vinho branco", "vinho tinto", "vinho rosé"],
            "es": ["sulfito", "dióxido de azufre", "vino", "vino blanco", "vino tinto", "vino rosado"],
        },
        "mustard": {
            "en": ["mustard", "mustard seed"],
            "pt": ["mostarda", "semente de mostarda"],
            "es": ["mostaza", "semilla de mostaza"],
        },
        "celery": {
            "en": ["celery", "celeriac", "celery seed", "celery salt"],
            "pt": ["aipo", "raiz de aipo", "semente de aipo", "sal de aipo"],
            "es": ["apio", "raíz de apio", "semilla de apio", "sal de apio"],
        },
        "lupin": {
            "en": ["lupin", "lupini"],
            "pt": ["tremoço", "tremoceiro"],
            "es": ["altramuz", "lupino"],
        },
        "molluscs": {
            "en": ["mollusc", "mollusk", "snail", "octopus"],
            "pt": ["molusco", "caracol", "polvo"],
            "es": ["molusco", "caracol", "pulpo"],
        },
    }

    def __init__(self):
        self._build_keyword_index()

    def _build_keyword_index(self):
        """Build keyword index for fast matching."""
        self.all_keywords = {}  # keyword → allergen_name
        for allergen, languages in self.ALLERGEN_GROUPS.items():
            for lang_keywords in languages.values():
                for keyword in lang_keywords:
                    keyword_lower = keyword.lower().strip()
                    self.all_keywords[keyword_lower] = allergen

    def detect_allergens(self, ingredient_text: str, language: str = "pt") -> Set[str]:
        """
        Detect allergens in ingredient text.
        Returns set of allergen names (lowercase, underscore-separated).
        """
        if not ingredient_text:
            return set()

        text_lower = ingredient_text.lower()
        detected = set()

        for keyword, allergen in self.all_keywords.items():
            # Match whole words with word boundaries
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                detected.add(allergen)

        return detected

    def get_allergen_display_name(self, allergen: str, language: str = "pt") -> str:
        """Get human-readable allergen name."""
        display_names = {
            "en": {
                "milk": "Milk", "eggs": "Eggs", "peanuts": "Peanuts", "tree_nuts": "Tree Nuts",
                "fish": "Fish", "shellfish": "Shellfish", "soy": "Soy", "wheat": "Wheat",
                "barley": "Barley", "rye": "Rye", "oats": "Oats", "sesame": "Sesame",
                "sulphites": "Sulphites", "mustard": "Mustard", "celery": "Celery",
                "lupin": "Lupin", "molluscs": "Molluscs",
            },
            "pt": {
                "milk": "Leite", "eggs": "Ovos", "peanuts": "Amendoim", "tree_nuts": "Frutos de Casca Rija",
                "fish": "Peixe", "shellfish": "Marisco", "soy": "Soja", "wheat": "Trigo",
                "barley": "Cevada", "rye": "Centeio", "oats": "Aveia", "sesame": "Gergelim",
                "sulphites": "Sulfitos", "mustard": "Mostarda", "celery": "Aipo",
                "lupin": "Tremoço", "molluscs": "Moluscos",
            },
            "es": {
                "milk": "Leche", "eggs": "Huevos", "peanuts": "Cacahuetes", "tree_nuts": "Frutos de Cáscara",
                "fish": "Pez", "shellfish": "Marisco", "soy": "Soja", "wheat": "Trigo",
                "barley": "Cebada", "rye": "Centeno", "oats": "Avena", "sesame": "Sésamo",
                "sulphites": "Sulfitos", "mustard": "Mostaza", "celery": "Apio",
                "lupin": "Altramuz", "molluscs": "Moluscos",
            },
        }
        return display_names.get(language, display_names["en"]).get(allergen, allergen)

    def get_allergen_risk_level(self, allergen: str) -> str:
        """
        Classify allergen risk level.
        Returns: "critical", "high", "medium"
        """
        critical = {"milk", "eggs", "peanuts", "tree_nuts", "fish", "shellfish", "soy", "wheat"}
        if allergen in critical:
            return "critical"
        return "high"  # All allergens are tracked as high priority


# ────────────────────────────────────────────────────────────────────────────
# PRODUCER & TRANSPORT EXTRACTOR
# ────────────────────────────────────────────────────────────────────────────

class IngredientAnalyzer:
    """Extract producer, origin, and transport info from ingredient data."""

    # Producer/brand patterns (PT/ES/EN)
    PRODUCER_PATTERNS = [
        r"(?:marca|brand|fabricante|manufacturer|productor|produtor)[\s:]*([a-záéíóúãõçñ\w\s&\-\.]+?)(?:\(|,|;|$)",
        r"^([A-Za-z][A-Za-z0-9\s&\-\.]{3,30})\s+(?:brand|marca|produced|producido|produzido)",
        r"(?:from|de|del|da|do)\s+([A-Za-z][A-Za-z0-9\s\-\.]{2,30})(?:\s+brand|\(|,|$)",
    ]

    # Origin/transport patterns
    ORIGIN_PATTERNS = [
        r"(?:origem|origin|procedencia|procedence|procedència)[\s:]*([a-zA-Z\s\-]+?)(?:\(|,|;|$)",
        r"(?:país|country|país|pais)[\s:]*([a-zA-Z\s\-]+?)(?:\(|,|;|$)",
    ]

    STORAGE_PATTERNS = [
        r"(?:armazenamento|storage|almacenamiento|transporte|transport)[\s:]*([a-záéíóúãõç\s\-,0-9]+?)(?:\(|;|$)",
        r"(?:manter|keep|guardar|store)[\s:]*(?:em|at|a|en)?\s*([a-záéíóúãõç\s\-,0-9]+?)(?:°C|\(|;|$)",
        r"(-?[\d]+\s*[-–]\s*[\d]+\s*°?C)",  # Temperature ranges
    ]

    @staticmethod
    def extract_producer(ingredient_text: Optional[str]) -> Optional[str]:
        """Extract producer/brand name from ingredient text."""
        if not ingredient_text:
            return None

        for pattern in IngredientAnalyzer.PRODUCER_PATTERNS:
            match = re.search(pattern, ingredient_text, re.IGNORECASE)
            if match:
                producer = match.group(1).strip()
                if len(producer) > 2 and len(producer) < 100:
                    return producer
        return None

    @staticmethod
    def extract_origin(ingredient_text: Optional[str]) -> Optional[str]:
        """Extract origin/country from ingredient text."""
        if not ingredient_text:
            return None

        for pattern in IngredientAnalyzer.ORIGIN_PATTERNS:
            match = re.search(pattern, ingredient_text, re.IGNORECASE)
            if match:
                origin = match.group(1).strip()
                if len(origin) > 2 and len(origin) < 50:
                    return origin
        return None

    @staticmethod
    def extract_storage_conditions(ingredient_text: Optional[str]) -> Optional[str]:
        """Extract storage/transport conditions from ingredient text."""
        if not ingredient_text:
            return None

        for pattern in IngredientAnalyzer.STORAGE_PATTERNS:
            match = re.search(pattern, ingredient_text, re.IGNORECASE)
            if match:
                conditions = match.group(1).strip()
                if len(conditions) > 2 and len(conditions) < 100:
                    return conditions
        return None


# ────────────────────────────────────────────────────────────────────────────
# ALLERGEN RISK ASSESSMENT
# ────────────────────────────────────────────────────────────────────────────

def assess_allergen_risk(allergens: set, quantity: float, unit: str) -> dict:
    """
    Assess risk level for ingredient based on allergen content and quantity.
    Returns: {"overall_risk": "critical|high|medium|low", "confidence": 0-1}
    """
    if not allergens:
        return {"overall_risk": "none", "confidence": 1.0}

    allergen_db = AllergenDatabase()

    # Check for critical allergens
    critical_allergens = {a for a in allergens if allergen_db.get_allergen_risk_level(a) == "critical"}

    # Risk depends on quantity
    trace_quantities = {"ml", "mg", "g"}  # Small units
    is_trace = unit.lower() in trace_quantities and quantity < 1

    if critical_allergens:
        if is_trace:
            overall_risk = "medium"  # Trace amounts
            confidence = 0.7
        else:
            overall_risk = "critical"
            confidence = 0.95
    else:
        overall_risk = "high"
        confidence = 0.85

    return {
        "overall_risk": overall_risk,
        "confidence": confidence,
        "critical_allergens": list(critical_allergens),
        "is_trace_amount": is_trace,
    }
