from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
from app.ml import initialize


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


@app.get("/health")
async def health():
    return {"status": "ok", "type": "local_ml"}
