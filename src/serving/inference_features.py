"""Build future features with the same pure functions used during training."""

from datetime import date, datetime, timedelta

import polars as pl

from src.processing import features


def build_inference_features(
    history: pl.DataFrame,
    weather: pl.DataFrame,
    station_master: pl.DataFrame,
    as_of: date,
    lag_hours: list[int],
    rolling_days: list[int],
    horizon_hours: int = 24,
) -> pl.DataFrame:
    """Return one future horizon for every station with complete recent history."""
    if not lag_hours or not rolling_days:
        raise ValueError("lag_hours and rolling_days must both be non-empty.")
    if horizon_hours > min(lag_hours):
        raise ValueError(
            f"horizon {horizon_hours}h > min lag {min(lag_hours)}h; "
            "future target values would enter lag features."
        )

    last_known = datetime(as_of.year, as_of.month, as_of.day, 23)
    history = history.filter(pl.col("hour") <= last_known)
    if history.is_empty():
        raise ValueError("History is empty through the requested as_of date.")

    needed_hours = max(max(lag_hours), max(rolling_days) * 24)
    history_start = last_known + timedelta(hours=1 - needed_hours)
    recent = history.filter(pl.col("hour") >= history_start)
    stations = history["station_id"].unique().sort()

    coverage = recent.group_by("station_id").agg(
        pl.len().alias("n_rows"),
        pl.col("hour").n_unique().alias("n_hours"),
        pl.col("hour").min().alias("first_hour"),
        pl.col("hour").max().alias("last_hour"),
    )
    expected = pl.DataFrame({"station_id": stations}).join(
        coverage,
        on="station_id",
        how="left",
    )
    incomplete = expected.filter(
        pl.col("n_rows").fill_null(0).ne(needed_hours)
        | pl.col("n_hours").fill_null(0).ne(needed_hours)
        | pl.col("first_hour").ne_missing(history_start)
        | pl.col("last_hour").ne_missing(last_known)
    )
    if not incomplete.is_empty():
        sample = incomplete["station_id"].head(5).to_list()
        raise ValueError(
            f"Recent history is incomplete for {incomplete.height} station(s), "
            f"including {sample}. Expected {needed_hours} continuous hourly rows "
            f"from {history_start} through {last_known}."
        )

    future_hours = pl.datetime_range(
        last_known + timedelta(hours=1),
        last_known + timedelta(hours=horizon_hours),
        "1h",
        eager=True,
    ).to_frame("hour")
    future = (
        pl.DataFrame({"station_id": stations})
        .join(future_hours, how="cross")
        .with_columns(
            pl.lit(None, dtype=recent.schema["trips"]).alias("trips")
        )
    )
    combined = pl.concat(
        [recent.select("station_id", "hour", "trips"), future],
        how="vertical",
    )

    frame = features.add_calendar_features(combined)
    frame = features.add_weather_features(frame, weather)
    frame = features.add_station_features(frame, station_master)
    frame = features.add_lag_features(frame, lag_hours, rolling_days)
    return frame.filter(pl.col("hour") > last_known)
