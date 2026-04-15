from __future__ import annotations

import json

import aiofiles
from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.exporter.xlsx_exporter import export_to_xlsx
from app.models.schema import NormalizedMenuSheet
from app.ml import get_normalizer, get_trainer
from app.parsers.csv_parser import extract_csv
from app.parsers.pdf_parser import extract_pdf
from app.parsers.xlsx_parser import extract_xlsx

router = APIRouter()

ALLOWED_EXTENSIONS = {"pdf", "csv", "xlsx"}
EXTENSION_TO_FORMAT = {"pdf": "pdf", "csv": "csv", "xlsx": "xlsx"}


class CorrectionMetadata(BaseModel):
    """Metadata for ML learning from corrections."""
    added_ingredients: int = 0
    deleted_ingredients: int = 0
    feedback_tagged: int = 0


class CorrectionSubmission(BaseModel):
    """Wrapper for correction data with ML metadata."""
    sheet: NormalizedMenuSheet
    metadata: CorrectionMetadata = CorrectionMetadata()


def _detect_format(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: .{ext}. Allowed: {ALLOWED_EXTENSIONS}",
        )
    return EXTENSION_TO_FORMAT[ext]


@router.post("/normalize", response_model=NormalizedMenuSheet, status_code=201)
async def normalize_sheet(file: UploadFile = File(...)):
    """
    Accept a PDF, CSV, or XLSX file. Parse, normalize with AI, return JSON.
    Also persists the result to storage/normalized/{id}.json.
    """
    # Validate file size
    content = await file.read()
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_file_size_mb}MB",
        )

    source_format = _detect_format(file.filename or "unknown.pdf")
    source_file = file.filename or "unknown"

    # Parse raw content based on format
    try:
        if source_format == "pdf":
            raw = await extract_pdf(content)
        elif source_format == "csv":
            raw = await extract_csv(content)
        elif source_format == "xlsx":
            raw = await extract_xlsx(content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Parse error: {str(e)}")

    # Local ML normalization
    try:
        normalizer = get_normalizer()
        sheet = await normalizer.normalize(
            raw, source_file=source_file, source_format=source_format
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Normalization error: {str(e)}")

    # Feed normalized sheet into learning systems (semantic graph, vocabulary)
    try:
        normalizer.rule_engine.learn_from_normalized_sheet(sheet)
    except Exception as e:
        # Log but don't fail - learning is non-critical
        print(f"Warning: Failed to learn from normalized sheet: {e}")

    # Persist to storage
    await _save_sheet(sheet)

    return sheet


@router.post("/normalize-bulk", status_code=201)
async def normalize_bulk(files: list[UploadFile] = File(...)):
    """
    Accept multiple PDF, CSV, or XLSX files. Normalize all and return summary.
    Feeds all sheets into learning systems for bulk vocabulary/semantic learning.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 files per request")

    results = {
        "total_files": len(files),
        "successful": 0,
        "failed": 0,
        "sheets": [],
        "errors": [],
    }

    normalizer = get_normalizer()

    for file in files:
        try:
            # Validate file size
            content = await file.read()
            if len(content) > settings.max_file_size_bytes:
                results["failed"] += 1
                results["errors"].append({
                    "file": file.filename,
                    "error": f"File too large. Max: {settings.max_file_size_mb}MB"
                })
                continue

            source_format = _detect_format(file.filename or "unknown.pdf")
            source_file = file.filename or "unknown"

            # Parse raw content based on format
            try:
                if source_format == "pdf":
                    raw = await extract_pdf(content)
                elif source_format == "csv":
                    raw = await extract_csv(content)
                elif source_format == "xlsx":
                    raw = await extract_xlsx(content)
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "file": file.filename,
                    "error": f"Parse error: {str(e)}"
                })
                continue

            # Normalize
            try:
                sheet = await normalizer.normalize(
                    raw, source_file=source_file, source_format=source_format
                )
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "file": file.filename,
                    "error": f"Normalization error: {str(e)}"
                })
                continue

            # Feed into learning systems
            try:
                normalizer.rule_engine.learn_from_normalized_sheet(sheet)
            except Exception as e:
                print(f"Warning: Failed to learn from {file.filename}: {e}")

            # Persist
            await _save_sheet(sheet)

            results["successful"] += 1
            results["sheets"].append({
                "id": sheet.id,
                "filename": file.filename,
                "name": sheet.name,
                "ingredients_count": len(sheet.ingredients),
                "allergens_detected": sum(len(ing.allergens) for ing in sheet.ingredients),
            })

        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "file": file.filename,
                "error": f"Unexpected error: {str(e)}"
            })

    # Get updated learning stats
    semantic_stats = normalizer.rule_engine.semantic_graph.get_graph_statistics()
    vocab_stats = normalizer.rule_engine.token_vocabulary.get_vocabulary_stats()

    results["learning_summary"] = {
        "semantic_graph": {
            "total_tokens": semantic_stats["total_tokens"],
            "ingredients_known": semantic_stats["ingredients"],
            "allergens_known": semantic_stats["allergens"],
            "total_relationships": semantic_stats["total_relationships"],
        },
        "vocabulary": {
            "total_tokens": vocab_stats["total_tokens"],
            "total_observations": vocab_stats["total_frequency"],
            "learned_merges": vocab_stats["total_merges"],
        },
    }

    return results


@router.get("/export/{sheet_id}")
async def export_sheet(sheet_id: str):
    """
    Load a previously normalized sheet and return as formatted XLSX download.
    """
    json_path = settings.storage_dir / f"{sheet_id}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Sheet not found: {sheet_id}")

    async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
        data = json.loads(await f.read())

    sheet = NormalizedMenuSheet(**data)
    xlsx_bytes = export_to_xlsx(sheet)

    filename = f"{sheet.name.replace(' ', '_')}_{sheet_id[:8]}.xlsx"

    return StreamingResponse(
        content=iter([xlsx_bytes]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/sheets/{sheet_id}", response_model=NormalizedMenuSheet)
async def get_sheet(sheet_id: str):
    """Retrieve a previously normalized sheet by ID."""
    json_path = settings.storage_dir / f"{sheet_id}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Not found")

    async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
        data = json.loads(await f.read())

    return NormalizedMenuSheet(**data)


async def _save_sheet(sheet: NormalizedMenuSheet) -> None:
    """Persist normalized sheet as JSON with raw_extraction included."""
    path = settings.storage_dir / f"{sheet.id}.json"
    # model_dump includes raw_extraction; use mode="json" for datetime serialization
    data = sheet.model_dump(mode="json")
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(data, ensure_ascii=False, indent=2))


@router.post("/corrections/{sheet_id}")
async def submit_correction(sheet_id: str, data: dict = Body(...)):
    """
    Submit a corrected version of a normalized sheet.
    Computes diff and records correction for model retraining.

    Accepts either:
    - A NormalizedMenuSheet (legacy format)
    - A wrapper with sheet + metadata (new format with add/delete/feedback)
    """
    # Extract sheet and metadata
    if "sheet" in data:
        # New format with metadata
        sheet_data = data["sheet"]
        metadata = data.get("metadata", {})
    else:
        # Legacy format: direct sheet
        sheet_data = data
        metadata = {}

    # Extract feedback/deletion markers BEFORE Pydantic strips them
    ingredient_metadata = {}
    if "ingredients" in sheet_data:
        for idx, ing in enumerate(sheet_data["ingredients"]):
            ingredient_metadata[idx] = {
                "marked_for_deletion": ing.get("_marked_for_deletion", False),
                "ml_feedback": ing.get("_ml_feedback", None),
            }

    # Construct Pydantic model (this strips extra attributes)
    corrected = NormalizedMenuSheet(**sheet_data)

    # Load original
    json_path = settings.storage_dir / f"{sheet_id}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Sheet not found")

    async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
        original_data = json.loads(await f.read())

    original = NormalizedMenuSheet(**original_data)

    # Filter out deleted ingredients before comparison
    # Use extracted metadata, not attributes on objects
    active_ingredients = [
        ing for idx, ing in enumerate(corrected.ingredients)
        if not ingredient_metadata.get(idx, {}).get("marked_for_deletion", False)
    ]

    corrected_active = NormalizedMenuSheet(
        **{
            **corrected.dict(),
            "ingredients": active_ingredients
        }
    )

    # Compute diff (includes feedback tags in metadata)
    diff = _compute_diff(original, corrected_active, metadata)
    if not diff:
        return {"status": "no_changes", "sheet_id": sheet_id}

    # Record correction
    trainer = get_trainer()
    diff_json = json.dumps(diff)
    correction_id = trainer.pattern_store.record_correction(
        sheet_id=sheet_id,
        restaurant_id=corrected.source_file.split("_")[0] if "_" in corrected.source_file else None,
        source_fmt=corrected.source_format,
        diff_json=diff_json,
    )

    # Save corrected version (without metadata fields)
    await _save_sheet(corrected_active)

    # Trigger async retraining
    await trainer.process_correction(
        correction_id=correction_id,
        diff_json=diff_json,
        source_fmt=corrected.source_format,
        restaurant_id=corrected.source_file.split("_")[0] if "_" in corrected.source_file else None,
    )

    # Learn from this correction (self-improving rules)
    try:
        normalizer = get_normalizer()
        feedback_data = {
            f"ingredient_{i}": {"feedback": ingredient_metadata.get(i, {}).get("ml_feedback", None)}
            for i in range(len(corrected.ingredients))
        }
        normalizer.rule_engine.learn_from_correction(original, corrected_active, feedback_data)
    except Exception as e:
        # Log but don't fail - learning is non-critical
        print(f"Warning: Failed to learn from correction: {e}")

    return {
        "status": "accepted",
        "sheet_id": sheet_id,
        "correction_id": correction_id,
        "changes": len(diff),
    }


@router.get("/status")
async def get_status():
    """Get system learning status."""
    from app.ml import get_pattern_store, get_normalizer

    store = get_pattern_store()
    normalizer = get_normalizer()

    # Get rule engine status
    rule_confidence = normalizer.rule_engine.get_learning_confidence()
    rules_summary = normalizer.rule_engine.get_rules_summary()

    # Get semantic graph status
    semantic_stats = normalizer.rule_engine.semantic_graph.get_graph_statistics()

    return {
        "correction_count": store.get_correction_count(),
        "pattern_count": store.get_pattern_count(),
        "restaurant_count": store.get_restaurant_count(),
        "model_metrics": {
            "column_classifier": {
                "is_trained": True,
            },
        },
        "learning_status": {
            "confidence": rule_confidence,
            "rules": rules_summary,
            "is_self_improving": rule_confidence["allergen_rules"] > 0,
        },
        "semantic_graph": {
            "total_tokens": semantic_stats["total_tokens"],
            "ingredients_known": semantic_stats["ingredients"],
            "allergens_known": semantic_stats["allergens"],
            "total_relationships": semantic_stats["total_relationships"],
            "avg_relationship_strength": semantic_stats["avg_relationship_strength"],
        },
        "vocabulary": {
            "total_tokens": normalizer.rule_engine.token_vocabulary.get_vocabulary_stats()["total_tokens"],
            "learned_merges": normalizer.rule_engine.token_vocabulary.get_vocabulary_stats()["total_merges"],
            "total_observations": normalizer.rule_engine.token_vocabulary.get_vocabulary_stats()["total_frequency"],
        },
    }


@router.get("/learning-rules")
async def get_learning_rules():
    """Get detailed information about learned rules."""
    from app.ml import get_normalizer

    normalizer = get_normalizer()
    return normalizer.rule_engine.get_rules_summary()


@router.get("/semantic/{ingredient}")
async def get_ingredient_semantic_profile(ingredient: str):
    """
    Get semantic analysis of an ingredient including:
    - Allergen profile
    - Related ingredients
    - Allergen predictions based on learned relationships
    - Ingredient family classification
    """
    from app.ml import get_normalizer

    normalizer = get_normalizer()
    return normalizer.rule_engine.get_ingredient_semantic_profile(ingredient)


@router.get("/semantic/correlations")
async def get_semantic_correlations():
    """
    Get comprehensive semantic analysis of all learned relationships:
    - Allergen correlations
    - Ingredient families
    - Graph statistics
    """
    from app.ml import get_normalizer

    normalizer = get_normalizer()
    return normalizer.rule_engine.analyze_semantic_correlations()


@router.get("/tokenization-activity")
async def get_tokenization_activity():
    """
    Get recent tokenization and semantic graph activity:
    - Recently created tokens
    - Recently created/updated relationships
    - Current graph snapshot with top tokens and relationships
    """
    from app.ml import get_normalizer

    normalizer = get_normalizer()
    return normalizer.rule_engine.get_tokenization_activity()


@router.get("/vocabulary-stats")
async def get_vocabulary_stats():
    """
    Get comprehensive vocabulary learning statistics:
    - Total tokens learned
    - Merge history
    - Merge candidates (tokens that appear together frequently)
    - Vocabulary coverage
    - Top tokens by frequency
    """
    from app.ml import get_normalizer

    normalizer = get_normalizer()
    return normalizer.rule_engine.get_vocabulary_stats()


@router.get("/sheets")
async def list_all_sheets():
    """
    List all normalized sheets in the database.
    Returns summary info for each sheet (id, name, ingredients count, allergens, etc.)
    """
    import json
    from pathlib import Path

    sheets = []
    storage_dir = settings.storage_dir

    if not storage_dir.exists():
        return {"sheets": [], "total": 0}

    for json_file in sorted(storage_dir.glob("*.json")):
        try:
            async with aiofiles.open(json_file, "r", encoding="utf-8") as f:
                data = json.loads(await f.read())

            sheet = NormalizedMenuSheet(**data)
            sheets.append({
                "id": sheet.id,
                "name": sheet.name,
                "category": sheet.category,
                "source_file": sheet.source_file,
                "source_format": sheet.source_format,
                "servings": sheet.servings,
                "ingredients_count": len(sheet.ingredients),
                "allergens_detected": sheet.allergen_ingredients_count,
                "critical_allergens": sheet.critical_allergens,
                "confidence_score": sheet.confidence_score,
                "normalized_at": sheet.normalized_at.isoformat(),
            })
        except Exception as e:
            # Skip files that can't be parsed
            print(f"Error reading {json_file}: {e}")
            continue

    return {"sheets": sheets, "total": len(sheets)}


@router.delete("/sheets/{sheet_id}")
async def delete_sheet(sheet_id: str):
    """Delete a normalized sheet from the database."""
    json_path = settings.storage_dir / f"{sheet_id}.json"

    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Sheet not found")

    try:
        json_path.unlink()
        return {"status": "deleted", "sheet_id": sheet_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete sheet: {str(e)}")


def _compute_diff(original: NormalizedMenuSheet, corrected: NormalizedMenuSheet, metadata: dict = None) -> dict:
    """
    Compute field-level differences between original and corrected.
    Returns dict of {field_path: {predicted, corrected}}.
    Also includes feedback and addition/deletion metadata for ML learning.
    """
    diff = {}
    metadata = metadata or {}

    # Compare scalar fields
    for field in [
        "name",
        "category",
        "country",
        "region",
        "continent",
        "servings",
        "prep_time_minutes",
        "cooking_time_minutes",
    ]:
        orig_val = getattr(original, field)
        corr_val = getattr(corrected, field)
        if orig_val != corr_val:
            diff[field] = {"predicted": orig_val, "corrected": corr_val}

    # Compare ingredients (tracking additions/deletions/feedback)
    if original.ingredients != corrected.ingredients:
        diff["ingredients"] = {
            "predicted": len(original.ingredients),
            "corrected": len(corrected.ingredients),
            "added": metadata.get("added_ingredients", 0),
            "deleted": metadata.get("deleted_ingredients", 0),
            "feedback_tags": metadata.get("feedback_tagged", 0),
        }

        # Track ingredient-level changes
        corrections_per_ing = []
        for i, (o_ing, c_ing) in enumerate(zip(original.ingredients, corrected.ingredients)):
            ing_diff = {}
            if o_ing.product_name != c_ing.product_name:
                ing_diff["product_name"] = {"predicted": o_ing.product_name, "corrected": c_ing.product_name}
            if o_ing.unit != c_ing.unit:
                ing_diff["unit"] = {"predicted": o_ing.unit, "corrected": c_ing.unit}
            if o_ing.quantity != c_ing.quantity:
                ing_diff["quantity"] = {"predicted": o_ing.quantity, "corrected": c_ing.quantity}
            if o_ing.unit_price != c_ing.unit_price:
                ing_diff["unit_price"] = {
                    "predicted": o_ing.unit_price,
                    "corrected": c_ing.unit_price,
                }
            # Track ML feedback
            feedback = getattr(c_ing, "_ml_feedback", None)
            if feedback:
                ing_diff["ml_feedback"] = feedback

            if ing_diff:
                corrections_per_ing.append({"index": i, "changes": ing_diff})

        if corrections_per_ing:
            diff["ingredient_corrections"] = corrections_per_ing

    # Compare steps
    if original.steps != corrected.steps:
        diff["steps"] = {
            "predicted": len(original.steps),
            "corrected": len(corrected.steps),
        }

    return diff
