"""Aggregate cleaned Divvy trips into station-hour data products."""

from datetime import timedelta

import polars as pl


def select_scope(clean: pl.LazyFrame, n: int) -> list[str]:
    """Select top-N stations by trip volume in the last 12 months."""
    max_started_at = clean.select(pl.col("started_at").max()).collect().item()
    cutoff = max_started_at - timedelta(days=365)

    top_stations = (
        clean.filter(pl.col("started_at") >= cutoff)
        .group_by("station_id")
        .agg(pl.len().alias("trips"))
        .sort("trips", descending=True)
        .head(n)
        .collect()
    )

    return top_stations["station_id"].to_list()


def aggregate_station_hour(clean: pl.LazyFrame, scope: list[str]) -> pl.DataFrame:
    """Count trips by station-hour within the selected station scope."""
    return (
        clean.filter(pl.col("station_id").is_in(scope))
        .with_columns(pl.col("started_at").dt.truncate("1h").alias("hour"))
        .group_by(["station_id", "hour"])
        .agg(pl.len().alias("trips"))
        .collect()
    )


def build_full_grid(counts: pl.DataFrame, scope: list[str]) -> pl.DataFrame:
    """Build full station-hour grid and fill missing station-hours with zero trips."""
    start_hour = counts["hour"].min()
    end_hour = counts["hour"].max()

    hours = pl.datetime_range(
        start_hour,
        end_hour,
        interval="1h",
        eager=True,
    ).to_frame("hour")

    station_grid = pl.DataFrame({"station_id": scope})
    grid = station_grid.join(hours, how="cross")

    return (
        grid.join(counts, on=["station_id", "hour"], how="left")
        .with_columns(pl.col("trips").fill_null(0).cast(pl.Int32))
        .sort(["station_id", "hour"])
    )


def trim_to_station_lifetime(grid: pl.DataFrame, clean: pl.LazyFrame) -> pl.DataFrame:
    """Remove grid rows before each station first appeared.

    This avoids teaching the model that a station had zero demand before it existed.
    """
    first_seen = (
        clean.group_by("station_id")
        .agg(pl.col("started_at").min().dt.truncate("1mo").alias("first_month"))
        .collect()
    )

    return (
        grid.join(first_seen, on="station_id", how="left")
        .filter(pl.col("hour") >= pl.col("first_month"))
        .drop("first_month")
        .sort(["station_id", "hour"])
    )


def build_station_master(clean: pl.LazyFrame) -> pl.DataFrame:
    """Build station-level reference table using canonical station IDs."""
    return (
        clean.group_by("station_id")
        .agg(
            pl.col("start_station_name").drop_nulls().mode().first().alias("name"),
            pl.col("start_lat").median().alias("lat"),
            pl.col("start_lng").median().alias("lng"),
            pl.len().alias("total_trips"),
            pl.col("started_at").min().dt.strftime("%Y-%m").alias("first_month"),
            pl.col("started_at").max().dt.strftime("%Y-%m").alias("last_month"),
        )
        .sort("total_trips", descending=True)
        .collect()
    )


def build_station_month_panel(clean: pl.LazyFrame) -> pl.DataFrame:
    """Build station-month panel split by member/casual trips."""
    panel = (
        clean.with_columns(pl.col("started_at").dt.strftime("%Y-%m").alias("month"))
        .group_by(["station_id", "month", "member_casual"])
        .agg(pl.len().alias("trips"))
        .collect()
        .pivot(
            on="member_casual",
            index=["station_id", "month"],
            values="trips",
            aggregate_function="sum",
        )
    )

    expected_cols = ["member", "casual"]
    for col in expected_cols:
        if col not in panel.columns:
            panel = panel.with_columns(pl.lit(0).alias(col))

    return (
        panel.rename(
            {
                "member": "member_trips",
                "casual": "casual_trips",
            }
        )
        .with_columns(
            pl.col("member_trips").fill_null(0).cast(pl.Int32),
            pl.col("casual_trips").fill_null(0).cast(pl.Int32),
        )
        .with_columns(
            (pl.col("member_trips") + pl.col("casual_trips")).alias("total_trips")
        )
        .sort(["station_id", "month"])
    )
