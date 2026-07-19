"""Compare replayed predictions with actual station-hour demand."""

from pathlib import Path

import polars as pl


def prediction_actual_table(
    predictions_dir: str | Path,
    actuals: pl.DataFrame,
) -> pl.DataFrame:
    """Return the Week 5 monitoring contract at station-hour granularity."""
    paths = sorted(Path(predictions_dir).glob("as_of=*.parquet"))
    if not paths:
        raise ValueError(f"No prediction partitions found in {predictions_dir}.")

    predictions = pl.read_parquet(paths)
    joined = predictions.join(
        actuals.select("station_id", "hour", "trips").rename(
            {"trips": "actual"}
        ),
        on=["station_id", "hour"],
        how="inner",
    )
    if joined.height != predictions.height:
        missing = predictions.height - joined.height
        print(f"Warning: {missing} predictions do not have actuals yet.")
    return joined.with_columns(
        (pl.col("prediction") - pl.col("actual")).abs().alias(
            "absolute_error"
        )
    )


def evaluate_predictions(
    predictions_dir: str | Path,
    actuals: pl.DataFrame,
) -> pl.DataFrame:
    detailed = prediction_actual_table(predictions_dir, actuals)
    if detailed.is_empty():
        raise ValueError("No predictions could be matched to actuals.")
    return (
        detailed.with_columns(pl.col("hour").dt.date().alias("target_date"))
        .group_by("as_of", "target_date")
        .agg(
            pl.col("absolute_error").mean().alias("mae"),
            pl.len().alias("n"),
        )
        .sort("as_of")
    )
