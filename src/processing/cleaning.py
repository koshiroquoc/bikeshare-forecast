"""Clean raw Divvy trips using config-driven filtering rules and station mapping."""

import polars as pl

DURATION_SECONDS = (pl.col("ended_at") - pl.col("started_at")).dt.total_seconds()


def _rules(cfg: dict) -> dict[str, pl.Expr]:
    """Return rule name -> survival condition.

    The same rule definitions are used for both removal reporting and actual cleaning.
    """
    scope = cfg["scope"]

    return {
        "missing_station": pl.col("start_station_id").is_not_null(),
        "too_short_or_negative": DURATION_SECONDS >= scope["min_trip_seconds"],
        "too_long": DURATION_SECONDS <= scope["max_trip_hours"] * 3600,
    }


def removal_report(trips: pl.LazyFrame, cfg: dict) -> dict[str, int]:
    """Count how many rows violate each cleaning rule.

    A single row can violate more than one rule, so these counts should not be summed
    to compute the final number of removed rows.
    """
    counts = trips.select(
        pl.len().alias("total"),
        *[(~expr).sum().alias(name) for name, expr in _rules(cfg).items()],
    ).collect()

    return {column: counts[column][0] for column in counts.columns}


def clean_trips(
    trips: pl.LazyFrame,
    cfg: dict,
    mapping: pl.DataFrame,
) -> pl.LazyFrame:
    """Apply cleaning rules and map raw station IDs to canonical station IDs."""
    survive = pl.all_horizontal(list(_rules(cfg).values()))

    return (
        trips.filter(survive)
        .join(
            mapping.lazy(),
            left_on="start_station_id",
            right_on="raw_station_id",
            how="left",
        )
        .with_columns(
            pl.coalesce(
                pl.col("canonical_station_id"),
                pl.col("start_station_id"),
            ).alias("station_id")
        )
        .drop("canonical_station_id")
    )
