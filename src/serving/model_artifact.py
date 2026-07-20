"""Self-describing model artifact used by batch serving."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from src.training.models import to_xy


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def save_artifact(
    model,
    categories: dict[str, list],
    features: list[str],
    params: dict,
    data_range: tuple[str, str],
    train_through: str,
    val_mae: float,
    out_dir: str | Path,
) -> Path:
    """Save the model and every piece of metadata required for prediction."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc)

    joblib.dump(model, out / "model.joblib")
    metadata = {
        "schema_version": 1,
        "features": features,
        "categories": categories,
        "params": params,
        "fitted_n_estimators": int(model.n_estimators_),
        "data_range": list(data_range),
        "train_through": train_through,
        "val_mae": float(val_mae),
        "git_commit": _git_hash(),
        "created_at": created_at.isoformat(),
        "model_version": created_at.strftime("%Y%m%d_%H%M%S_%f"),
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return out


def load_artifact(model_dir: str | Path) -> tuple:
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    metadata = load_metadata(model_dir)
    return model, metadata


def load_metadata(model_dir: str | Path) -> dict:
    """Load artifact metadata without loading the LightGBM model."""
    model_dir = Path(model_dir)
    return json.loads(
        (model_dir / "metadata.json").read_text(encoding="utf-8")
    )


def predict_frame(model, metadata: dict, df: pl.DataFrame) -> pl.Series:
    """Predict with the exact feature order and categories used at training."""
    x, _ = to_xy(df, metadata["features"], metadata["categories"])
    predictions = model.predict(x)
    return pl.Series(np.maximum(predictions, 0.0))
