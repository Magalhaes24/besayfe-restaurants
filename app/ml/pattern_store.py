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
