"""
SQLite-backed pattern store for restaurant menu normalization.
Maintains learned patterns from corrections and provides per-restaurant overrides.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import re


class PatternStore:
    """Manages SQLite database of learned patterns and corrections."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # column_mappings: training data (raw header → canonical field)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS column_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_name TEXT NOT NULL,
                canonical TEXT NOT NULL,
                source_fmt TEXT NOT NULL,
                restaurant TEXT,
                confidence REAL DEFAULT 1.0,
                origin TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )

        # unit_vocabulary: unit normalization (raw → "KG", "L", "G", "ML", "UN")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS unit_vocabulary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_unit TEXT NOT NULL,
                canonical TEXT NOT NULL,
                restaurant TEXT,
                origin TEXT DEFAULT 'heuristic',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(raw_unit, restaurant)
            )
            """
        )

        # restaurant_profiles: per-restaurant metadata
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS restaurant_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                restaurant_id TEXT NOT NULL UNIQUE,
                display_name TEXT,
                language TEXT DEFAULT 'pt',
                total_sheets INTEGER DEFAULT 0,
                total_corrections INTEGER DEFAULT 0,
                last_seen_at TEXT
            )
            """
        )

        # corrections: raw diff records from user feedback
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sheet_id TEXT NOT NULL,
                restaurant_id TEXT,
                source_fmt TEXT,
                diff_json TEXT NOT NULL,
                applied INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )

        # pdf_field_patterns: regex patterns for PDF scalar extraction
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pdf_field_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                field_name TEXT NOT NULL,
                pattern TEXT NOT NULL,
                language TEXT DEFAULT 'pt',
                restaurant TEXT,
                priority INTEGER DEFAULT 0,
                hit_count INTEGER DEFAULT 0,
                origin TEXT DEFAULT 'heuristic',
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )

        # model_metrics: accuracy tracking over time
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT NOT NULL,
                accuracy REAL,
                n_samples INTEGER,
                n_corrections INTEGER DEFAULT 0,
                model_version INTEGER DEFAULT 0,
                recorded_at TEXT DEFAULT (datetime('now'))
            )
            """
        )

        # vocabulary_tokens: persistent token vocabulary (BPE-inspired)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS vocabulary_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL UNIQUE,
                token_type TEXT DEFAULT 'ingredient',
                frequency INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.5,
                language TEXT DEFAULT 'pt',
                created_at TEXT DEFAULT (datetime('now')),
                last_used TEXT DEFAULT (datetime('now'))
            )
            """
        )

        # vocabulary_merges: learned token merges
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS vocabulary_merges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_a TEXT NOT NULL,
                token_b TEXT NOT NULL,
                merged_token TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.5,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(token_a, token_b)
            )
            """
        )

        # vocabulary_cooccurrence: token pair frequency matrix
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS vocabulary_cooccurrence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_a TEXT NOT NULL,
                token_b TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(token_a, token_b)
            )
            """
        )

        # semantic_tokens: semantic graph tokens (ingredients, allergens, origins, etc.)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                value TEXT NOT NULL,
                token_type TEXT NOT NULL,
                language TEXT DEFAULT 'pt',
                confidence REAL DEFAULT 1.0,
                created_at TEXT DEFAULT (datetime('now')),
                last_used TEXT DEFAULT (datetime('now')),
                UNIQUE(value, token_type)
            )
            """
        )

        # semantic_relationships: relationships between semantic tokens
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_value TEXT NOT NULL,
                source_type TEXT NOT NULL,
                target_value TEXT NOT NULL,
                target_type TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                strength REAL DEFAULT 0.5,
                evidence_count INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now')),
                last_updated TEXT DEFAULT (datetime('now')),
                UNIQUE(source_value, source_type, target_value, target_type, relation_type)
            )
            """
        )

        # ingredient_families: computed ingredient families based on allergen profiles
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ingredient_families (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                family_key TEXT NOT NULL UNIQUE,
                allergen_profile TEXT NOT NULL,
                ingredients_json TEXT NOT NULL,
                member_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                last_updated TEXT DEFAULT (datetime('now'))
            )
            """
        )

        conn.commit()
        conn.close()

        # Seed with heuristic patterns
        self._seed_heuristics()

    def _seed_heuristics(self):
        """Seed database with cold-start heuristic patterns."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if already seeded
        cursor.execute("SELECT COUNT(*) FROM unit_vocabulary")
        if cursor.fetchone()[0] > 0:
            conn.close()
            return

        # Unit vocabulary heuristics (PT/ES/EN)
        unit_mappings = [
            ("kg", "KG"),
            ("kilogram", "KG"),
            ("quilograma", "KG"),
            ("l", "L"),
            ("litre", "L"),
            ("litro", "L"),
            ("ml", "ML"),
            ("mililitre", "ML"),
            ("mililitro", "ML"),
            ("g", "G"),
            ("gram", "G"),
            ("grama", "G"),
            ("un", "UN"),
            ("unit", "UN"),
            ("unidad", "UN"),
            ("pcs", "UN"),
            ("piece", "UN"),
            ("pieza", "UN"),
        ]

        for raw, canonical in unit_mappings:
            cursor.execute(
                "INSERT OR IGNORE INTO unit_vocabulary (raw_unit, canonical, origin) VALUES (?, ?, 'heuristic')",
                (raw.lower(), canonical),
            )

        # PDF field pattern heuristics (Portuguese)
        pdf_patterns = [
            ("servings", r"por[çc][õo]es?\s*[:\-]?\s*(\d+)", "pt", 10),
            ("prep_time_minutes", r"tempo\s+de\s+prepara[çc][ãa]o\s*[:\-]?\s*(\d+)\s*min", "pt", 10),
            ("cooking_time_minutes", r"tempo\s+de\s+con(?:f|cc)e[çc][ãa]o\s*[:\-]?\s*(\d+)\s*min", "pt", 10),
            ("category", r"categoria\s*[:\-]?\s*([a-záéíóúãõç\s]+?)(?:\n|$)", "pt", 5),
            ("servings", r"raciones?\s*[:\-]?\s*(\d+)", "es", 10),
            ("prep_time_minutes", r"tiempo\s+de\s+preparaci[óo]n\s*[:\-]?\s*(\d+)\s*min", "es", 10),
            ("cooking_time_minutes", r"tiempo\s+de\s+cocci[óo]n\s*[:\-]?\s*(\d+)\s*min", "es", 10),
        ]

        for field, pattern, lang, priority in pdf_patterns:
            cursor.execute(
                "INSERT INTO pdf_field_patterns (field_name, pattern, language, priority, origin) VALUES (?, ?, ?, ?, 'heuristic')",
                (field, pattern, lang, priority),
            )

        conn.commit()
        conn.close()

    def detect_restaurant_id(self, source_file: str, raw_text: str = "") -> str:
        """
        Detect restaurant ID from filename prefix or header scan.
        Returns slugified ID.
        """
        # Try to extract from filename (e.g., "restaurantex_2_c_sheet.pdf" → "restaurantex")
        filename = Path(source_file).stem.lower()
        match = re.match(r"([a-z0-9_]+?)[\s\-_]?[\d\-._]", filename)
        if match:
            return match.group(1).replace(" ", "_").replace("-", "_")

        # Fallback: generic
        return "generic"

    def normalize_unit(self, raw_unit: str, restaurant: Optional[str] = None) -> str:
        """
        Lookup unit normalization: restaurant-specific → global → built-in.
        """
        if not raw_unit:
            return ""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        normalized = raw_unit.strip().upper()

        # Restaurant-specific
        if restaurant:
            cursor.execute(
                "SELECT canonical FROM unit_vocabulary WHERE raw_unit = LOWER(?) AND restaurant = ?",
                (raw_unit, restaurant),
            )
            result = cursor.fetchone()
            if result:
                conn.close()
                return result[0]

        # Global
        cursor.execute(
            "SELECT canonical FROM unit_vocabulary WHERE raw_unit = LOWER(?) AND restaurant IS NULL",
            (raw_unit,),
        )
        result = cursor.fetchone()
        if result:
            conn.close()
            return result[0]

        conn.close()
        return normalized

    def get_restaurant_column_overrides(self, restaurant_id: str) -> dict:
        """
        Get per-restaurant column name overrides.
        Returns {raw_name: canonical}.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT raw_name, canonical FROM column_mappings WHERE restaurant = ? ORDER BY confidence DESC",
            (restaurant_id,),
        )

        overrides = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return overrides

    def add_column_mapping(
        self,
        raw_name: str,
        canonical: str,
        source_fmt: str,
        restaurant: Optional[str] = None,
        confidence: float = 1.0,
        origin: str = "heuristic",
    ):
        """Record a column name mapping."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO column_mappings
            (raw_name, canonical, source_fmt, restaurant, confidence, origin)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (raw_name, canonical, source_fmt, restaurant, confidence, origin),
        )

        conn.commit()
        conn.close()

    def add_unit_mapping(
        self,
        raw_unit: str,
        canonical: str,
        restaurant: Optional[str] = None,
        origin: str = "heuristic",
    ):
        """Record a unit normalization mapping."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO unit_vocabulary
            (raw_unit, canonical, restaurant, origin)
            VALUES (LOWER(?), ?, ?, ?)
            """,
            (raw_unit, canonical, restaurant, origin),
        )

        conn.commit()
        conn.close()

    def record_correction(
        self,
        sheet_id: str,
        restaurant_id: Optional[str],
        source_fmt: str,
        diff_json: str,
    ) -> int:
        """Record a user correction. Returns correction ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO corrections (sheet_id, restaurant_id, source_fmt, diff_json)
            VALUES (?, ?, ?, ?)
            """,
            (sheet_id, restaurant_id, source_fmt, diff_json),
        )

        correction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return correction_id

    def get_pending_corrections(self) -> list[dict]:
        """Get all unapplied corrections."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, sheet_id, restaurant_id, source_fmt, diff_json
            FROM corrections WHERE applied = 0
            ORDER BY created_at
            """
        )

        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "sheet_id": row[1],
                "restaurant_id": row[2],
                "source_fmt": row[3],
                "diff_json": row[4],
            })

        conn.close()
        return results

    def mark_correction_applied(self, correction_id: int):
        """Mark a correction as applied."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("UPDATE corrections SET applied = 1 WHERE id = ?", (correction_id,))

        conn.commit()
        conn.close()

    def get_all_column_examples(self, canonical: str) -> list[str]:
        """Get all raw names that map to a canonical field."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT DISTINCT raw_name FROM column_mappings WHERE canonical = ? ORDER BY confidence DESC",
            (canonical,),
        )

        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return results

    def get_correction_count(self) -> int:
        """Total corrections recorded."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM corrections")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_pattern_count(self) -> int:
        """Total patterns (mappings + units)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM column_mappings")
        mappings = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM unit_vocabulary")
        units = cursor.fetchone()[0]
        conn.close()
        return mappings + units

    def get_restaurant_count(self) -> int:
        """Total unique restaurants with patterns."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM restaurant_profiles")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def update_restaurant_profile(
        self,
        restaurant_id: str,
        display_name: Optional[str] = None,
        language: str = "pt",
    ):
        """Create or update restaurant profile."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO restaurant_profiles
            (restaurant_id, display_name, language, last_seen_at)
            VALUES (?, ?, ?, ?)
            """,
            (restaurant_id, display_name or restaurant_id, language, datetime.utcnow().isoformat()),
        )

        conn.commit()
        conn.close()

    def get_pdf_field_patterns(
        self,
        field_name: str,
        language: str = "pt",
        restaurant: Optional[str] = None,
    ) -> list[str]:
        """Get regex patterns for a PDF field (restaurant-specific first)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        patterns = []

        # Restaurant-specific
        if restaurant:
            cursor.execute(
                """
                SELECT pattern FROM pdf_field_patterns
                WHERE field_name = ? AND language = ? AND restaurant = ?
                ORDER BY priority DESC, hit_count DESC
                """,
                (field_name, language, restaurant),
            )
            patterns.extend([row[0] for row in cursor.fetchall()])

        # Global
        cursor.execute(
            """
            SELECT pattern FROM pdf_field_patterns
            WHERE field_name = ? AND language = ? AND restaurant IS NULL
            ORDER BY priority DESC, hit_count DESC
            """,
            (field_name, language),
        )
        patterns.extend([row[0] for row in cursor.fetchall()])

        conn.close()
        return patterns

    def record_pattern_hit(self, pattern_id: int):
        """Increment hit count for a pattern (used to track effectiveness)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("UPDATE pdf_field_patterns SET hit_count = hit_count + 1 WHERE id = ?", (pattern_id,))

        conn.commit()
        conn.close()

    # --- Vocabulary Persistence (BPE-inspired Token Learning) ---

    def save_vocabulary_token(
        self,
        text: str,
        token_type: str = "ingredient",
        frequency: int = 1,
        confidence: float = 0.5,
        language: str = "pt",
    ):
        """Save or update a vocabulary token."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO vocabulary_tokens (text, token_type, frequency, confidence, language, last_used)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(text) DO UPDATE SET
                frequency = frequency + 1,
                confidence = MAX(confidence, excluded.confidence),
                last_used = datetime('now')
            """,
            (text.lower().strip(), token_type, frequency, confidence, language),
        )

        conn.commit()
        conn.close()

    def save_vocabulary_merge(
        self,
        token_a: str,
        token_b: str,
        merged_token: str,
        frequency: int = 1,
        confidence: float = 0.5,
    ):
        """Save or update a learned token merge."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        a_lower = token_a.lower().strip()
        b_lower = token_b.lower().strip()
        merged_lower = merged_token.lower().strip()

        cursor.execute(
            """
            INSERT INTO vocabulary_merges (token_a, token_b, merged_token, frequency, confidence)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(token_a, token_b) DO UPDATE SET
                frequency = frequency + 1,
                confidence = MAX(confidence, excluded.confidence)
            """,
            (a_lower, b_lower, merged_lower, frequency, confidence),
        )

        conn.commit()
        conn.close()

    def save_cooccurrence(self, token_a: str, token_b: str, count: int = 1):
        """Save or update token pair cooccurrence count."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        a_lower = token_a.lower().strip()
        b_lower = token_b.lower().strip()

        cursor.execute(
            """
            INSERT INTO vocabulary_cooccurrence (token_a, token_b, count)
            VALUES (?, ?, ?)
            ON CONFLICT(token_a, token_b) DO UPDATE SET
                count = count + excluded.count
            """,
            (a_lower, b_lower, count),
        )

        conn.commit()
        conn.close()

    def load_vocabulary(self) -> dict:
        """Load entire vocabulary from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Load tokens
        cursor.execute("SELECT * FROM vocabulary_tokens")
        tokens = {}
        for row in cursor.fetchall():
            tokens[row[1]] = {  # text is key
                "text": row[1],
                "type": row[2],
                "frequency": row[3],
                "confidence": row[4],
                "language": row[5],
                "created_at": row[6],
                "last_used": row[7],
            }

        # Load merges
        cursor.execute("SELECT * FROM vocabulary_merges")
        merges = []
        for row in cursor.fetchall():
            merges.append({
                "a": row[1],
                "b": row[2],
                "merged": row[3],
                "frequency": row[4],
                "confidence": row[5],
                "created_at": row[6],
            })

        # Load cooccurrence matrix
        cursor.execute("SELECT * FROM vocabulary_cooccurrence")
        cooccurrence = {}
        for row in cursor.fetchall():
            cooccurrence[f"{row[1]}:{row[2]}"] = row[3]

        conn.close()

        return {
            "vocabulary": tokens,
            "merges": merges,
            "cooccurrence_matrix": cooccurrence,
        }

    def get_vocabulary_stats(self) -> dict:
        """Get comprehensive vocabulary statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM vocabulary_tokens")
        total_tokens = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(frequency) FROM vocabulary_tokens")
        total_frequency = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM vocabulary_merges")
        total_merges = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(confidence) FROM vocabulary_merges")
        avg_confidence = cursor.fetchone()[0] or 0.0

        # Top tokens
        cursor.execute(
            "SELECT text, token_type, frequency, confidence, last_used FROM vocabulary_tokens ORDER BY frequency DESC LIMIT 10"
        )
        top_tokens = [
            {
                "text": row[0],
                "type": row[1],
                "frequency": row[2],
                "confidence": round(row[3], 2),
                "last_used": row[4],
            }
            for row in cursor.fetchall()
        ]

        # Recent merges
        cursor.execute(
            "SELECT token_a, token_b, merged_token, frequency, confidence, created_at FROM vocabulary_merges ORDER BY created_at DESC LIMIT 5"
        )
        recent_merges = [
            {
                "a": row[0],
                "b": row[1],
                "merged": row[2],
                "frequency": row[3],
                "confidence": round(row[4], 2),
                "created_at": row[5],
            }
            for row in cursor.fetchall()
        ]

        conn.close()

        return {
            "total_tokens": total_tokens,
            "total_frequency": total_frequency,
            "avg_token_frequency": total_frequency / max(1, total_tokens),
            "total_merges": total_merges,
            "avg_merge_confidence": round(avg_confidence, 2),
            "top_tokens": top_tokens,
            "recent_merges": recent_merges,
        }

    def save_semantic_token(self, value: str, token_type: str, language: str = "pt", confidence: float = 1.0):
        """Save or update a semantic token in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO semantic_tokens (value, token_type, language, confidence)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(value, token_type) DO UPDATE SET
                  confidence = MAX(confidence, ?),
                  last_used = datetime('now')
                """,
                (value.lower().strip(), token_type, language, confidence, confidence)
            )
            conn.commit()
        finally:
            conn.close()

    def save_semantic_relationship(self, source_val: str, source_type: str, target_val: str, target_type: str, relation_type: str, strength: float = 0.5):
        """Save or update a semantic relationship in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO semantic_relationships
                (source_value, source_type, target_value, target_type, relation_type, strength, evidence_count)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(source_value, source_type, target_value, target_type, relation_type) DO UPDATE SET
                  strength = ?,
                  evidence_count = evidence_count + 1,
                  last_updated = datetime('now')
                """,
                (source_val.lower().strip(), source_type, target_val.lower().strip(), target_type, relation_type, strength, strength)
            )
            conn.commit()
        finally:
            conn.close()

    def load_semantic_graph(self) -> dict:
        """Load all semantic tokens and relationships from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            # Load tokens
            cursor.execute("SELECT value, token_type, language, confidence, created_at, last_used FROM semantic_tokens")
            tokens = {}
            for row in cursor.fetchall():
                key = (row[0], row[1])
                tokens[key] = {
                    "value": row[0],
                    "type": row[1],
                    "language": row[2],
                    "confidence": row[3],
                    "created_at": row[4],
                    "last_used": row[5],
                }

            # Load relationships
            cursor.execute(
                """
                SELECT source_value, source_type, target_value, target_type, relation_type, strength, evidence_count, created_at
                FROM semantic_relationships
                """
            )
            relationships = []
            for row in cursor.fetchall():
                relationships.append({
                    "source_value": row[0],
                    "source_type": row[1],
                    "target_value": row[2],
                    "target_type": row[3],
                    "relation_type": row[4],
                    "strength": row[5],
                    "evidence_count": row[6],
                    "created_at": row[7],
                })

            # Load families
            cursor.execute(
                """
                SELECT family_key, allergen_profile, ingredients_json, member_count
                FROM ingredient_families
                """
            )
            families = {}
            for row in cursor.fetchall():
                families[row[0]] = {
                    "allergen_profile": row[1],
                    "ingredients": row[2],
                    "member_count": row[3],
                }

            return {
                "tokens": tokens,
                "relationships": relationships,
                "families": families,
            }
        finally:
            conn.close()

    def update_ingredient_family(self, family_key: str, allergen_profile: str, ingredients_json: str, member_count: int):
        """Save or update an ingredient family."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO ingredient_families (family_key, allergen_profile, ingredients_json, member_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(family_key) DO UPDATE SET
                  allergen_profile = ?,
                  ingredients_json = ?,
                  member_count = ?,
                  last_updated = datetime('now')
                """,
                (family_key, allergen_profile, ingredients_json, member_count, allergen_profile, ingredients_json, member_count)
            )
            conn.commit()
        finally:
            conn.close()
