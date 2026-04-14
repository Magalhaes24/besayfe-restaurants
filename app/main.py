from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.api.debug_routes import debug_router
from app.config import settings
from app.ml import initialize

STATIC_DIR = Path(__file__).parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure storage directories exist and initialize ML
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.ml_dir.mkdir(parents=True, exist_ok=True)
    initialize(settings.ml_dir)
    yield
    # Shutdown: nothing to clean up


app = FastAPI(
    title="Restaurant Menu Sheet Normalizer",
    description="AI-powered normalization of restaurant technical sheets (fichas técnicas)",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")
app.include_router(debug_router, prefix="/api/v1")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_ui():
    """Serve the debug UI."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "ok", "ui": "not found — place static/index.html"}


@app.get("/health")
async def health():
    return {"status": "ok", "type": "local_ml"}
