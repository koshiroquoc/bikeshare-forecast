"""Tests for src/training/events.py."""

from datetime import date, datetime

import polars as pl

from src.training.events import add_event_feature, load_event_days


def _grid(start: datetime, end: datetime) -> pl.DataFrame:
    hours = pl.datetime_range(start, end, interval="1h", eager=True)

    return (
        pl.DataFrame({"hour": hours})
        .join(pl.DataFrame({"station_id": ["a", "b"]}), how="cross")
        .with_columns(pl.lit(1.0).alias("trips"))
    )


def _days(pairs: list[tuple[date, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [pair[0] for pair in pairs],
            "event_name": [pair[1] for pair in pairs],
        }
    )


def test_row_count_unchanged() -> None:
    """Event joins must not add or remove rows."""
    grid = _grid(datetime(2025, 7, 30), datetime(2025, 8, 5, 23))
    out = add_event_feature(grid, _days([(date(2025, 8, 1), "lolla")]))

    assert out.height == grid.height


def test_flag_only_on_event_days() -> None:
    grid = _grid(datetime(2025, 7, 31), datetime(2025, 8, 2, 23))
    out = add_event_feature(grid, _days([(date(2025, 8, 1), "lolla")]))

    by_day = (
        out.with_columns(pl.col("hour").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("is_major_event").mean())
    )

    flagged = {str(row["date"]): row["is_major_event"] for row in by_day.to_dicts()}

    assert flagged == {
        "2025-07-31": 0.0,
        "2025-08-01": 1.0,
        "2025-08-02": 0.0,
    }


def test_duplicate_event_dates_do_not_duplicate_rows() -> None:
    """Multiple events on the same date should still produce one date flag."""
    grid = _grid(datetime(2025, 8, 1), datetime(2025, 8, 1, 23))

    event_days = _days(
        [
            (date(2025, 8, 1), "event_1"),
            (date(2025, 8, 1), "event_2"),
        ]
    ).unique(subset=["date"], keep="first")

    out = add_event_feature(grid, event_days)

    assert out.height == grid.height


def test_load_event_days_from_real_csv() -> None:
    """The real CSV should load as unique event dates."""
    days = load_event_days("data/reference/chicago_events.csv")

    assert days["date"].dtype == pl.Date
    assert days["date"].n_unique() == days.height
