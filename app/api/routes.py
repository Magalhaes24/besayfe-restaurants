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

    # Persist to storage
    await _save_sheet(sheet)

    return sheet


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
        corrected = NormalizedMenuSheet(**data["sheet"])
        metadata = data.get("metadata", {})
    else:
        # Legacy format: direct sheet
        corrected = NormalizedMenuSheet(**data)
        metadata = {}

    # Load original
    json_path = settings.storage_dir / f"{sheet_id}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Sheet not found")

    async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
        original_data = json.loads(await f.read())

    original = NormalizedMenuSheet(**original_data)

    # Filter out deleted ingredients before comparison
    corrected_active = NormalizedMenuSheet(
        **{
            **corrected.dict(),
            "ingredients": [
                i for i in corrected.ingredients
                if not getattr(i, "_marked_for_deletion", False)
            ]
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
        feedback_data = {f"ingredient_{i}": {"feedback": getattr(ing, "_ml_feedback", None)}
                        for i, ing in enumerate(corrected.ingredients)}
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
    }


@router.get("/learning-rules")
async def get_learning_rules():
    """Get detailed information about learned rules."""
    from app.ml import get_normalizer

    normalizer = get_normalizer()
    return normalizer.rule_engine.get_rules_summary()


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
