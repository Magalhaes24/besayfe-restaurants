"""
Local ML normalizer module. Manages singletons for pattern store, classifiers, and trainer.
"""

from __future__ import annotations

from pathlib import Path

from app.ml.pattern_store import PatternStore

_pattern_store: PatternStore | None = None
_normalizer = None
_trainer = None


def initialize(ml_dir: Path) -> None:
    """Initialize ML singletons."""
    global _pattern_store
    db_path = ml_dir / "patterns.db"
    _pattern_store = PatternStore(db_path)


def get_pattern_store() -> PatternStore:
    """Get the pattern store singleton."""
    global _pattern_store
    if _pattern_store is None:
        raise RuntimeError("ML system not initialized. Call initialize(ml_dir) first.")
    return _pattern_store


def get_normalizer():
    """Get the local normalizer singleton."""
    global _normalizer
    if _normalizer is None:
        from app.ml.normalizer import LocalNormalizer

        _normalizer = LocalNormalizer(get_pattern_store())
    return _normalizer


def get_trainer():
    """Get the trainer singleton."""
    global _trainer
    if _trainer is None:
        from app.ml.trainer import Trainer

        _trainer = Trainer(get_pattern_store())
    return _trainer
