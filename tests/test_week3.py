"""Week 3 model tests.

These tests check model properties rather than exact metric values. LightGBM
can vary slightly across versions and machines, so tests should verify that the
model learns, predicts valid counts, respects time-based validation, and handles
unknown stations without crashing.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from src.processing import features
from src.training import backtest
from src.training.models import LGBMPredictor, time_based_split

FAST_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "lag_24",
    "lag_48",
    "lag_168",
    "roll_mean_7d",
]

FAST_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_child_samples": 5,
    "random_state": 42,
}


def make_learnable(
    n_days: int = 200,
    stations: tuple[str, ...] = ("A", "B", "C"),
    seed: int = 0,
) -> pl.DataFrame:
    """Create synthetic station-hour demand with learnable structure.

    Demand has:
    - station-level scale
    - morning and evening commute peaks
    - lower weekend demand
    - Poisson noise

    A model that uses calendar and lag features should beat a one-week seasonal
    copy because it can learn the smooth expected pattern instead of copying one
    noisy historical observation.
    """
    rng = np.random.default_rng(seed)

    hours = pl.datetime_range(
        datetime(2025, 1, 1),
        datetime(2025, 1, 1) + timedelta(days=n_days, hours=-1),
        interval="1h",
        eager=True,
    )

    hour_of_day = np.array([hour.hour for hour in hours])
    is_weekend = np.array([hour.weekday() >= 5 for hour in hours])

    morning_peak = 2.5 * np.exp(-((hour_of_day - 8) ** 2) / 8)
    evening_peak = 3.0 * np.exp(-((hour_of_day - 17) ** 2) / 8)
    hour_shape = 0.4 + morning_peak + evening_peak

    frames = []

    for station_index, station_id in enumerate(stations):
        station_scale = 3.0 + 1.5 * station_index
        weekend_factor = np.where(is_weekend, 0.65, 1.0)
        lam = station_scale * hour_shape * weekend_factor
        trips = rng.poisson(lam).astype(np.int32)

        frames.append(
            pl.DataFrame(
                {
                    "station_id": [station_id] * len(hours),
                    "hour": hours,
                    "trips": trips,
                }
            )
        )

    grid = pl.concat(frames).sort("station_id", "hour")
    df = features.add_calendar_features(grid)

    return features.add_lag_features(
        df,
        lag_hours=[24, 48, 168],
        rolling_days=[7],
    )


def test_time_based_split_validation_is_after_fit() -> None:
    """The validation split must be time-based, never random."""
    df = make_learnable(n_days=120)

    fit_df, valid_df = time_based_split(df)

    assert fit_df.height > 0
    assert valid_df.height > 0
    assert fit_df["hour"].max() < valid_df["hour"].min()
    assert valid_df["hour"].min().day == 1


def test_lgbm_predictor_smoke_shape_and_non_negative() -> None:
    """LGBM should fit and produce one non-negative prediction per eval row."""
    df = make_learnable(n_days=180)
    eval_month = backtest.eval_months(df, 1)[0]

    month_df = df.with_columns(pl.col("hour").dt.strftime("%Y-%m").alias("_month"))
    train_df = month_df.filter(pl.col("_month") < eval_month)
    eval_df = month_df.filter(pl.col("_month") == eval_month)

    predictor = LGBMPredictor(
        FAST_FEATURES,
        FAST_PARAMS,
        name="lgbm_test",
    )

    preds = predictor.predict(train_df, eval_df)

    assert len(preds) == eval_df.height
    assert preds.null_count() == 0
    assert preds.min() >= 0


def test_lgbm_learns_pattern_better_than_seasonal_naive() -> None:
    """On learnable synthetic data, LGBM should beat seasonal naive."""
    df = make_learnable(n_days=220)
    eval_month = backtest.eval_months(df, 1)[0]

    month_df = df.with_columns(pl.col("hour").dt.strftime("%Y-%m").alias("_month"))
    train_df = month_df.filter(pl.col("_month") < eval_month)
    eval_df = month_df.filter(pl.col("_month") == eval_month)

    naive = backtest.SeasonalNaive()
    lgbm = LGBMPredictor(
        FAST_FEATURES,
        FAST_PARAMS,
        name="lgbm_test",
    )

    naive_preds = naive.predict(train_df, eval_df)
    lgbm_preds = lgbm.predict(train_df, eval_df)

    naive_mae = backtest.mae(eval_df["trips"], naive_preds)
    lgbm_mae = backtest.mae(eval_df["trips"], lgbm_preds)

    assert lgbm_mae < naive_mae


def test_unknown_eval_station_does_not_crash() -> None:
    """A station present only in eval should become an unknown category safely."""
    hours_train = pl.datetime_range(
        datetime(2025, 1, 1),
        datetime(2025, 2, 28, 23),
        interval="1h",
        eager=True,
    )

    hours_eval = pl.datetime_range(
        datetime(2025, 3, 1),
        datetime(2025, 3, 2, 23),
        interval="1h",
        eager=True,
    )

    train_a = pl.DataFrame(
        {
            "station_id": ["A"] * len(hours_train),
            "hour": hours_train,
            "trips": [2] * len(hours_train),
        }
    )

    train_b = pl.DataFrame(
        {
            "station_id": ["B"] * len(hours_train),
            "hour": hours_train,
            "trips": [5] * len(hours_train),
        }
    )

    train_df = pl.concat([train_a, train_b])

    eval_df = pl.DataFrame(
        {
            "station_id": ["C"] * len(hours_eval),
            "hour": hours_eval,
            "trips": [3] * len(hours_eval),
        }
    )

    train_df = features.add_calendar_features(train_df)
    eval_df = features.add_calendar_features(eval_df)

    predictor = LGBMPredictor(
        ["station_id", "hour_of_day", "day_of_week", "is_weekend"],
        FAST_PARAMS,
        name="lgbm_unknown_station_test",
    )

    preds = predictor.predict(train_df, eval_df)

    assert len(preds) == eval_df.height
    assert preds.null_count() == 0
    assert preds.min() >= 0
