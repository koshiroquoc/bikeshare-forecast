"""Week 2 backtest tests."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from src.processing import features
from src.training import backtest

LAG_HOURS = [24, 48, 168]
ROLLING_DAYS = [7]


def make_periodic_features(n_days: int = 120) -> pl.DataFrame:
    """Create a perfectly weekly-periodic series.

    Seasonal naive should achieve MAE = 0 on this data.
    """
    hours = pl.datetime_range(
        datetime(2025, 1, 1),
        datetime(2025, 1, 1) + timedelta(days=n_days, hours=-1),
        interval="1h",
        eager=True,
    )

    grid = pl.DataFrame(
        {
            "station_id": ["A"] * len(hours),
            "hour": hours,
            "trips": [(i % 168) % 7 for i in range(len(hours))],
        }
    ).sort(["station_id", "hour"])

    df = features.add_calendar_features(grid)
    return features.add_lag_features(df, LAG_HOURS, ROLLING_DAYS)


def _grid_ending_at(end: datetime) -> pl.DataFrame:
    hours = pl.datetime_range(
        datetime(2025, 1, 1),
        end,
        interval="1h",
        eager=True,
    )

    return pl.DataFrame(
        {
            "station_id": ["A"] * len(hours),
            "hour": hours,
            "trips": [0] * len(hours),
        }
    )


def test_eval_months_keeps_full_last_month() -> None:
    months = backtest.eval_months(
        _grid_ending_at(datetime(2025, 5, 31, 23)),
        n_windows=2,
    )

    assert months == ["2025-04", "2025-05"]


def test_eval_months_drops_partial_last_month() -> None:
    months = backtest.eval_months(
        _grid_ending_at(datetime(2025, 5, 17, 10)),
        n_windows=2,
    )

    assert months == ["2025-03", "2025-04"]


def test_seasonal_naive_perfect_on_periodic_series() -> None:
    df = make_periodic_features()
    results = backtest.run_backtest(
        df,
        [backtest.SeasonalNaive()],
        n_windows=2,
    )

    assert results.height == 2
    assert results["mae"].max() == 0.0


def test_historical_mean_perfect_on_constant_series() -> None:
    df = make_periodic_features().with_columns(pl.lit(5).alias("trips"))

    results = backtest.run_backtest(
        df,
        [backtest.HistoricalMean()],
        n_windows=2,
    )

    assert results["mae"].max() == pytest.approx(0.0)


def test_summarize_has_mase() -> None:
    df = make_periodic_features()
    results = backtest.run_backtest(
        df,
        [backtest.SeasonalNaive(), backtest.HistoricalMean()],
        n_windows=2,
    )

    summary = backtest.summarize(results)

    assert "mase_mean" in summary.columns
