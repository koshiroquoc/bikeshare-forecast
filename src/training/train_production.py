"""Train a production artifact through an explicit data cutoff."""

import argparse
from datetime import date, datetime

import polars as pl
import yaml

from src.serving.model_artifact import save_artifact
from src.training.models import fit_lgbm


def _end_of_day(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 23)


def train_production(
    features_df: pl.DataFrame,
    cfg: dict,
    out_dir: str = "models/current",
    train_through: date | None = None,
):
    """Select tree count on a validation month, then refit through the cutoff."""
    if train_through is not None:
        cutoff = _end_of_day(train_through)
        features_df = features_df.filter(pl.col("hour") <= cutoff)
    if features_df.is_empty():
        raise ValueError("No training rows are available through the requested cutoff.")

    actual_cutoff = features_df["hour"].max()
    features = cfg["model"]["features"]
    params = cfg["model"]["params"]
    model, categories, val_mae = fit_lgbm(
        features_df,
        features,
        params,
        refit_full=True,
    )
    data_range = (
        str(features_df["hour"].min()),
        str(features_df["hour"].max()),
    )
    path = save_artifact(
        model=model,
        categories=categories,
        features=features,
        params=params,
        data_range=data_range,
        train_through=str(actual_cutoff),
        val_mae=val_mae,
        out_dir=out_dir,
    )
    print(
        f"Model -> {path} | validation MAE: {val_mae:.3f} | "
        f"refit through: {actual_cutoff} | trees: {model.n_estimators_}"
    )
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--out", default="models/current")
    parser.add_argument(
        "--train-through",
        help="Optional YYYY-MM-DD cutoff. Required for honest historical replay.",
    )
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    features_table = pl.read_parquet("data/processed/features.parquet")
    cutoff_date = (
        date.fromisoformat(args.train_through) if args.train_through else None
    )
    train_production(features_table, config, args.out, cutoff_date)
