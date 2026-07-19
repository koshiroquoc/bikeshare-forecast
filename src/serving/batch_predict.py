"""Batch prediction: artifact -> future features -> partitioned prediction store."""

import argparse
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import yaml

from src.serving.inference_features import build_inference_features
from src.serving.model_artifact import load_artifact, predict_frame


def _artifact_train_date(metadata: dict) -> date:
    return datetime.fromisoformat(metadata["train_through"]).date()


def run_batch_predict(
    as_of: date,
    history: pl.DataFrame,
    weather: pl.DataFrame,
    station_master: pl.DataFrame,
    cfg: dict,
    model_dir: str | Path = "models/current",
    out_dir: str | Path = "data/predictions",
    force: bool = False,
) -> Path:
    """Create one idempotent partition for the day after ``as_of``."""
    model, metadata = load_artifact(model_dir)
    train_date = _artifact_train_date(metadata)
    if train_date > as_of:
        raise ValueError(
            f"Artifact was trained through {train_date}, after as_of={as_of}. "
            "Historical replay would leak future data. Train a cutoff artifact first."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"as_of={as_of}.parquet"
    if out_path.exists() and not force:
        existing_versions = (
            pl.read_parquet(out_path, columns=["model_version"])
            ["model_version"]
            .unique()
            .to_list()
        )
        if existing_versions == [metadata["model_version"]]:
            print(f"{out_path} already exists for this model; skipped")
            return out_path
        raise ValueError(
            f"{out_path} belongs to model version(s) {existing_versions}, not "
            f"{metadata['model_version']}. Use --force only if replacement is intended."
        )

    inference_frame = build_inference_features(
        history=history,
        weather=weather,
        station_master=station_master,
        as_of=as_of,
        lag_hours=cfg["features"]["lag_hours"],
        rolling_days=cfg["features"]["rolling_days"],
    )
    predictions = predict_frame(model, metadata, inference_frame)
    created_at = datetime.now(timezone.utc).isoformat()
    table = inference_frame.select("station_id", "hour").with_columns(
        pl.lit(as_of).alias("as_of"),
        predictions.alias("prediction"),
        pl.lit(metadata["model_version"]).alias("model_version"),
        pl.lit(created_at).alias("created_at"),
    ).select(
        "as_of",
        "station_id",
        "hour",
        "prediction",
        "model_version",
        "created_at",
    )

    temporary_path = out_path.with_suffix(".tmp.parquet")
    table.write_parquet(temporary_path)
    temporary_path.replace(out_path)
    print(
        f"{as_of}: {table.height:,} predictions "
        f"({table['station_id'].n_unique()} stations x 24h) -> {out_path}"
    )
    return out_path


def load_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load historical replay inputs from existing Week 1–2 data products."""
    history = pl.read_parquet(
        "data/processed/features.parquet",
        columns=["station_id", "hour", "trips"],
    )
    weather = pl.read_parquet("data/raw/weather/historical/*.parquet")
    station_master = pl.read_parquet("data/processed/station_master.parquet")
    return history, weather, station_master


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model-dir", default="models/current")
    parser.add_argument("--out-dir", default="data/predictions")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    history_df, weather_df, station_df = load_inputs()
    run_batch_predict(
        as_of=date.fromisoformat(args.as_of),
        history=history_df,
        weather=weather_df,
        station_master=station_df,
        cfg=config,
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        force=args.force,
    )
