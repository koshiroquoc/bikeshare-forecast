"""Chicago major-event feature: is_major_event.

The event CSV lives at data/reference/chicago_events.csv and contains:
start_date,end_date,event_name

No leakage: major-event calendars are public before the event happens, so this
feature is available for both training and future serving.

Implementation choice:
The CSV can contain multi-day events. We expand each event range into one row
per event day, then keep one row per date. This makes the final join a simple
left join on date and prevents row multiplication.
"""

import polars as pl


def load_event_days(path: str = "data/reference/chicago_events.csv") -> pl.DataFrame:
    """Load event CSV and return one unique row per event date."""
    events = pl.read_csv(path, try_parse_dates=True)

    return (
        events.with_columns(
            pl.date_ranges(
                pl.col("start_date"),
                pl.col("end_date"),
            ).alias("date")
        )
        .explode("date", empty_as_null=True)
        .select("date", "event_name")
        .unique(subset=["date"], keep="first")
    )


def add_event_feature(
    df: pl.DataFrame,
    event_days: pl.DataFrame,
) -> pl.DataFrame:
    """Add is_major_event without changing row count."""
    out = (
        df.with_columns(pl.col("hour").dt.date().alias("_event_date"))
        .join(
            event_days.rename({"date": "_event_date"}),
            on="_event_date",
            how="left",
        )
        .with_columns(
            pl.col("event_name").is_not_null().cast(pl.Int8).alias("is_major_event")
        )
        .drop("_event_date", "event_name")
    )

    assert out.height == df.height, (
        "Event join changed row count. Check that event_days is unique by date."
    )

    return out
