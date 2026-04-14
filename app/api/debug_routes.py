"""
Debug API routes — extra introspection endpoints for the debug UI.
Not intended for production use.
"""

from __future__ import annotations

import json
import sqlite3

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from app.config import settings
from app.ml import get_normalizer, get_pattern_store
from app.ml.column_classifier import SYNONYM_DICT

debug_router = APIRouter(prefix="/debug", tags=["debug"])


@debug_router.post("/classify-headers")
async def classify_headers(headers: list[str] = Body(...)):
    """
    Classify a list of raw column headers.
    Returns per-header classification with method used (synonym / ml / ignore).
    """
    normalizer = get_normalizer()
    classifier = normalizer.column_classifier

    results = []
    for header in headers:
        header_lower = header.strip().lower()

        # Determine method
        if not header_lower:
            results.append({"header": header, "canonical": "IGNORE", "method": "empty"})
            continue

        if header_lower in SYNONYM_DICT:
            results.append({
                "header": header,
                "canonical": SYNONYM_DICT[header_lower],
                "method": "synonym_exact",
            })
            continue

        substring_match = None
        for syn, canonical in SYNONYM_DICT.items():
            if syn in header_lower:
                substring_match = canonical
                break

        if substring_match:
            results.append({
                "header": header,
                "canonical": substring_match,
                "method": "synonym_substring",
            })
            continue

        if classifier.is_trained:
            canonical = classifier.classify_single(header)
            results.append({
                "header": header,
                "canonical": canonical,
                "method": "ml_model" if canonical != "IGNORE" else "ml_fallback_ignore",
            })
        else:
            results.append({
                "header": header,
                "canonical": "IGNORE",
                "method": "no_model",
            })

    return results


@debug_router.get("/patterns")
async def get_patterns():
    """Return all patterns currently in the pattern store."""
    store = get_pattern_store()
    conn = sqlite3.connect(store.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM column_mappings ORDER BY created_at DESC LIMIT 200")
    column_mappings = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM unit_vocabulary ORDER BY created_at DESC LIMIT 200")
    unit_vocabulary = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM pdf_field_patterns ORDER BY priority DESC, hit_count DESC LIMIT 100")
    pdf_patterns = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM corrections ORDER BY created_at DESC LIMIT 50")
    corrections = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM restaurant_profiles ORDER BY last_seen_at DESC")
    restaurants = [dict(r) for r in cursor.fetchall()]

    conn.close()

    return {
        "column_mappings": column_mappings,
        "unit_vocabulary": unit_vocabulary,
        "pdf_field_patterns": pdf_patterns,
        "corrections": corrections,
        "restaurants": restaurants,
        "synonym_dict": {k: v for k, v in SYNONYM_DICT.items()},
    }


@debug_router.get("/ml-status")
async def get_ml_status():
    """Return ML model status details."""
    normalizer = get_normalizer()
    classifier = normalizer.column_classifier
    store = get_pattern_store()

    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM column_mappings")
    n_mappings = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM corrections WHERE applied = 1")
    n_applied = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM corrections WHERE applied = 0")
    n_pending = cursor.fetchone()[0]
    conn.close()

    return {
        "column_classifier": {
            "is_trained": classifier.is_trained,
            "model_path": str(classifier.model_path),
            "model_exists": classifier.model_path.exists(),
            "training_examples": n_mappings,
        },
        "pattern_store": {
            "db_path": str(store.db_path),
            "total_corrections": store.get_correction_count(),
            "applied_corrections": n_applied,
            "pending_corrections": n_pending,
            "total_patterns": store.get_pattern_count(),
            "total_restaurants": store.get_restaurant_count(),
        },
    }


@debug_router.get("/sheets")
async def list_sheets():
    """List all normalized sheets in storage."""
    normalized_dir = settings.storage_dir
    sheets = []

    if normalized_dir.exists():
        for json_file in sorted(normalized_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sheets.append({
                    "id": data.get("id", json_file.stem),
                    "name": data.get("name", "Unknown"),
                    "category": data.get("category", ""),
                    "source_file": data.get("source_file", ""),
                    "source_format": data.get("source_format", ""),
                    "confidence_score": data.get("confidence_score", 0),
                    "total_cost": data.get("total_cost", 0),
                    "servings": data.get("servings", 0),
                    "n_ingredients": len(data.get("ingredients", [])),
                    "n_steps": len(data.get("steps", [])),
                    "normalized_at": data.get("normalized_at", ""),
                })
            except Exception:
                pass

    return {"sheets": sheets, "total": len(sheets)}
