from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    storage_dir: Path = Path("storage/normalized")
    ml_dir: Path = Path("storage/ml")
    max_file_size_mb: int = 20
    min_classifier_confidence: float = 0.5
    min_training_examples_per_class: int = 5
    retrain_after_n_corrections: int = 1

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def db_path(self) -> Path:
        return self.ml_dir / "patterns.db"

    @property
    def models_dir(self) -> Path:
        return self.ml_dir / "models"


settings = Settings()
