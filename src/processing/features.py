"""Feature engineering for station-hour bikeshare demand.

The same feature functions should be used for both training and serving to avoid
train/serve skew.

Leakage rule:
For a 24-hour forecast horizon, any feature derived from the target `trips`
must only use information at least 24 hours before the prediction hour.
"""

import math

import holidays
import polars as pl


def add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add time-based features that do not depend on the target."""
    min_year = df["hour"].dt.year().min()
    max_year = df["hour"].dt.year().max()

    il_holidays = set(
        holidays.US(subdiv="IL", years=range(min_year, max_year + 1)).keys()
    )
    two_pi = 2 * math.pi

    return df.with_columns(
        pl.col("hour").dt.hour().alias("hour_of_day"),
        pl.col("hour").dt.weekday().alias("day_of_week"),
        pl.col("hour").dt.month().alias("month"),
        (pl.col("hour").dt.weekday() >= 6).alias("is_weekend"),
        pl.col("hour").dt.date().is_in(sorted(il_holidays)).alias("is_holiday"),
    ).with_columns(
        (pl.col("hour_of_day") * two_pi / 24).sin().alias("hour_sin"),
        (pl.col("hour_of_day") * two_pi / 24).cos().alias("hour_cos"),
        (pl.col("month") * two_pi / 12).sin().alias("month_sin"),
        (pl.col("month") * two_pi / 12).cos().alias("month_cos"),
    )


def add_weather_features(df: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    """Join hourly weather features onto station-hour demand rows.

    Historical training uses actual weather. Future serving will use forecast weather.
    The schema is intentionally the same.
    """
    weather_hourly = weather.rename({"timestamp": "hour"})
    out = df.join(weather_hourly, on="hour", how="left")

    match_rate = out["temperature_2m"].is_not_null().mean()
    if match_rate < 0.95:
        raise ValueError(
            f"Weather join matched only {match_rate:.1%} of rows. "
            "This suggests a timezone mismatch or missing weather coverage."
        )

    return out


def add_station_features(
    df: pl.DataFrame, station_master: pl.DataFrame
) -> pl.DataFrame:
    """Join station-level static features."""
    return df.join(
        station_master.select(["station_id", "lat", "lng"]),
        on="station_id",
        how="left",
    )


def add_lag_features(
    df: pl.DataFrame,
    lag_hours: list[int],
    rolling_days: list[int],
) -> pl.DataFrame:
    """Add leak-safe lag and same-hour rolling features.

    For a 24-hour horizon, all target-derived features must use lag >= 24h.

    Rolling K days means:
    mean of the same hour over the previous K days:
    t-24h, t-48h, ..., t-(24*K)h.

    This is not a normal rolling window over the previous K*24 rows, because that
    would include information from t-1h through t-23h and leak future information.
    """
    if min(lag_hours) < 24:
        raise ValueError(
            "All lag features must be at least 24 hours for a 24h horizon."
        )

    df = df.sort(["station_id", "hour"])

    lag_exprs = [
        pl.col("trips").shift(h).over("station_id").alias(f"lag_{h}") for h in lag_hours
    ]

    rolling_exprs = [
        pl.mean_horizontal(
            [
                pl.col("trips").shift(24 * day).over("station_id")
                for day in range(1, k + 1)
            ]
        ).alias(f"roll_mean_{k}d")
        for k in rolling_days
    ]

    return df.with_columns(lag_exprs + rolling_exprs)
