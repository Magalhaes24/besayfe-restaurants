from __future__ import annotations

import json

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

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
async def submit_correction(sheet_id: str, corrected: NormalizedMenuSheet):
    """
    Submit a corrected version of a normalized sheet.
    Computes diff and records correction for model retraining.
    """
    # Load original
    json_path = settings.storage_dir / f"{sheet_id}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Sheet not found")

    async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
        original_data = json.loads(await f.read())

    original = NormalizedMenuSheet(**original_data)

    # Compute diff
    diff = _compute_diff(original, corrected)
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

    # Save corrected version
    await _save_sheet(corrected)

    # Trigger async retraining
    await trainer.process_correction(
        correction_id=correction_id,
        diff_json=diff_json,
        source_fmt=corrected.source_format,
        restaurant_id=corrected.source_file.split("_")[0] if "_" in corrected.source_file else None,
    )

    return {
        "status": "accepted",
        "sheet_id": sheet_id,
        "correction_id": correction_id,
        "changes": len(diff),
    }


@router.get("/status")
async def get_status():
    """Get system learning status."""
    from app.ml import get_pattern_store

    store = get_pattern_store()
    return {
        "correction_count": store.get_correction_count(),
        "pattern_count": store.get_pattern_count(),
        "restaurant_count": store.get_restaurant_count(),
        "model_metrics": {
            "column_classifier": {
                "is_trained": True,  # TODO: track actual status
            },
        },
    }


def _compute_diff(original: NormalizedMenuSheet, corrected: NormalizedMenuSheet) -> dict:
    """
    Compute field-level differences between original and corrected.
    Returns dict of {field_path: {predicted, corrected}}.
    """
    diff = {}

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

    # Compare ingredients
    if original.ingredients != corrected.ingredients:
        diff["ingredients"] = {
            "predicted": len(original.ingredients),
            "corrected": len(corrected.ingredients),
        }
        for i, (o_ing, c_ing) in enumerate(zip(original.ingredients, corrected.ingredients)):
            if o_ing != c_ing:
                if o_ing.unit != c_ing.unit:
                    diff[f"ingredients.{i}.unit"] = {"predicted": o_ing.unit, "corrected": c_ing.unit}
                if o_ing.quantity != c_ing.quantity:
                    diff[f"ingredients.{i}.quantity"] = {"predicted": o_ing.quantity, "corrected": c_ing.quantity}
                if o_ing.unit_price != c_ing.unit_price:
                    diff[f"ingredients.{i}.unit_price"] = {
                        "predicted": o_ing.unit_price,
                        "corrected": c_ing.unit_price,
                    }

    # Compare steps
    if original.steps != corrected.steps:
        diff["steps"] = {
            "predicted": len(original.steps),
            "corrected": len(corrected.steps),
        }

    return diff
