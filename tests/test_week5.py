"""Week 5 gates: every decision path must be exercised."""

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.monitoring import checks
from src.processing import features
from src.serving.model_artifact import save_artifact
from src.training import promote
from src.training.models import fit_lgbm

LAGS = [24, 48, 168]
ROLLS = [7]
MODEL_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "temperature_2m",
    "station_id",
    "lag_24",
    "lag_168",
    "roll_mean_7d",
]
PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "num_leaves": 31,
}
CONFIG = {
    "model": {
        "features": MODEL_FEATURES,
        "params": PARAMS,
    },
    "features": {
        "lag_hours": LAGS,
        "rolling_days": ROLLS,
    },
}


def make_features(n_days: int = 120, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    hours = pl.datetime_range(
        datetime(2025, 1, 1),
        datetime(2025, 1, 1)
        + timedelta(days=n_days, hours=-1),
        "1h",
        eager=True,
    )
    hour_of_day = np.array([hour.hour for hour in hours])
    shape = (
        0.3
        + 2.0 * np.exp(-((hour_of_day - 8) ** 2) / 6)
        + 2.4 * np.exp(-((hour_of_day - 17) ** 2) / 6)
    )
    temperature = (
        10
        + 8 * np.sin(np.arange(len(hours)) / 400)
        + rng.normal(0, 1.5, len(hours))
    )
    weather = pl.DataFrame(
        {
            "timestamp": hours,
            "temperature_2m": temperature,
            "precipitation": 0.0,
            "snowfall": 0.0,
            "wind_speed_10m": 8.0,
            "relative_humidity_2m": 70.0,
        }
    )
    grid = pl.concat(
        [
            pl.DataFrame(
                {
                    "station_id": [station] * len(hours),
                    "hour": hours,
                    "trips": rng.poisson(scale * shape).astype(np.int32),
                }
            )
            for station, scale in [("A", 5), ("B", 2)]
        ]
    ).sort("station_id", "hour")
    frame = features.add_calendar_features(grid)
    frame = features.add_weather_features(frame, weather)
    return features.add_lag_features(frame, LAGS, ROLLS)


def _daily(maes: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "target_date": [
                date(2025, 4, day + 1)
                for day in range(len(maes))
            ],
            "mae": maes,
            "n": [48] * len(maes),
        }
    )


def test_performance_gate_passes_within_tolerance():
    result = checks.performance_gate(
        _daily([1.0, 1.1, 1.05]),
        baseline_mae=1.0,
        tolerance=0.20,
    )
    assert result["passed"]
    assert result["mean_mae"] == 1.05


def test_performance_gate_fails_on_degradation():
    result = checks.performance_gate(
        _daily([1.5, 1.6, 1.7]),
        baseline_mae=1.0,
        tolerance=0.20,
    )
    assert not result["passed"]
    assert result["mean_mae"] > result["threshold"]


def _drift_frames(shift: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    reference = pl.DataFrame(
        {
            "temperature_2m": rng.normal(10, 5, 3000),
            "lag_24": rng.poisson(3, 3000).astype(np.float64),
            "lag_168": rng.poisson(3, 3000).astype(np.float64),
        }
    )
    current = pl.DataFrame(
        {
            "temperature_2m": rng.normal(10 + shift, 5, 800),
            "lag_24": rng.poisson(3 + shift, 800).astype(np.float64),
            "lag_168": rng.poisson(3 + shift, 800).astype(np.float64),
        }
    )
    return reference, current


def test_drift_gate_passes_on_same_distribution():
    reference, current = _drift_frames(0.0)
    result = checks.drift_gate(reference, current)
    assert result["passed"], result


def test_drift_gate_fails_and_writes_report(tmp_path):
    reference, current = _drift_frames(12.0)
    report_path = tmp_path / "drift.html"
    result = checks.drift_gate(
        reference,
        current,
        out_html=report_path,
    )
    assert not result["passed"], result
    assert report_path.exists()


def _save_current(
    frame: pl.DataFrame,
    out_dir,
    cutoff: datetime,
    *,
    bad: bool = False,
):
    training = frame.filter(pl.col("hour") <= cutoff)
    if bad:
        rng = np.random.default_rng(9)
        training = training.with_columns(
            pl.Series(
                "trips",
                rng.permutation(training["trips"].to_numpy()),
            )
        )
    model, categories, val_mae = fit_lgbm(
        training,
        MODEL_FEATURES,
        PARAMS,
        refit_full=True,
    )
    save_artifact(
        model=model,
        categories=categories,
        features=MODEL_FEATURES,
        params=PARAMS,
        data_range=(
            str(training["hour"].min()),
            str(training["hour"].max()),
        ),
        train_through=str(training["hour"].max()),
        val_mae=val_mae,
        out_dir=out_dir,
    )


def test_promotion_promotes_over_bad_champion(tmp_path):
    frame = make_features()
    current_dir = tmp_path / "current"
    archive_dir = tmp_path / "archive"
    _save_current(
        frame,
        current_dir,
        datetime(2025, 3, 31, 23),
        bad=True,
    )
    old_metadata = (current_dir / "metadata.json").read_text()

    decision = promote.promote_if_better(
        frame,
        CONFIG,
        current_dir=current_dir,
        archive_dir=archive_dir,
        min_improvement=0.02,
    )
    assert decision["promoted"]
    assert decision["eval_month"] == "2025-04"
    assert decision["mae_candidate"] < decision["mae_current"]
    assert (current_dir / "metadata.json").read_text() != old_metadata
    assert any(archive_dir.iterdir())


def test_promotion_keeps_good_champion_without_required_gain(tmp_path):
    frame = make_features()
    current_dir = tmp_path / "current"
    _save_current(
        frame,
        current_dir,
        datetime(2025, 3, 31, 23),
    )
    old_metadata = (current_dir / "metadata.json").read_text()

    decision = promote.promote_if_better(
        frame,
        CONFIG,
        current_dir=current_dir,
        archive_dir=tmp_path / "archive",
        min_improvement=0.02,
    )
    assert not decision["promoted"]
    assert decision["status"] == "kept"
    assert (current_dir / "metadata.json").read_text() == old_metadata
    assert not (tmp_path / "archive").exists()


def test_promotion_waits_without_unseen_full_month(tmp_path):
    frame = make_features()
    current_dir = tmp_path / "current"
    _save_current(
        frame,
        current_dir,
        datetime(2025, 4, 30, 23),
    )
    decision = promote.promote_if_better(
        frame,
        CONFIG,
        current_dir=current_dir,
        archive_dir=tmp_path / "archive",
    )
    assert not decision["promoted"]
    assert decision["status"] == "waiting_for_unseen_month"
    assert decision["eval_month"] is None


def test_monitoring_rejects_prediction_model_mismatch(tmp_path):
    frame = make_features()
    current_dir = tmp_path / "current"
    _save_current(
        frame,
        current_dir,
        datetime(2025, 3, 31, 23),
    )
    actual_row = frame.filter(
        pl.col("hour") == datetime(2025, 4, 1)
    ).head(1)
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    actual_row.select("station_id", "hour").with_columns(
        pl.lit(date(2025, 3, 31)).alias("as_of"),
        pl.lit(0.0).alias("prediction"),
        pl.lit("wrong-version").alias("model_version"),
        pl.lit("now").alias("created_at"),
    ).select(
        "as_of",
        "station_id",
        "hour",
        "prediction",
        "model_version",
        "created_at",
    ).write_parquet(predictions_dir / "as_of=2025-03-31.parquet")

    with pytest.raises(ValueError, match="do not match"):
        checks.run_monitoring(
            frame,
            predictions_dir,
            current_dir,
            report_dir=tmp_path / "reports",
        )
