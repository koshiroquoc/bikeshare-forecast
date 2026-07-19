"""Week 4 serving tests, including train/serve consistency and replay safety."""

import json
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from src.processing import features
from src.serving import batch_predict, inference_features
from src.serving.api import create_app
from src.serving.evaluate import evaluate_predictions, prediction_actual_table
from src.serving.model_artifact import (
    load_artifact,
    predict_frame,
    save_artifact,
)
from src.training.models import fit_lgbm, to_xy
from src.training.train_production import train_production

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
    "n_estimators": 80,
    "learning_rate": 0.1,
    "num_leaves": 31,
}


def make_world(n_days: int = 70, seed: int = 0):
    rng = np.random.default_rng(seed)
    hours = pl.datetime_range(
        datetime(2025, 3, 1),
        datetime(2025, 3, 1) + timedelta(days=n_days, hours=-1),
        "1h",
        eager=True,
    )
    hour_of_day = np.array([hour.hour for hour in hours])
    shape = (
        0.3
        + 2.0 * np.exp(-((hour_of_day - 8) ** 2) / 6)
        + 2.4 * np.exp(-((hour_of_day - 17) ** 2) / 6)
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
    weather = pl.DataFrame(
        {
            "timestamp": hours,
            "temperature_2m": (
                12
                + 8 * np.sin(np.arange(len(hours)) / 300)
                + rng.normal(0, 1, len(hours))
            ),
            "precipitation": 0.0,
            "snowfall": 0.0,
            "wind_speed_10m": 8.0,
            "relative_humidity_2m": 70.0,
        }
    )
    station_master = pl.DataFrame(
        {
            "station_id": ["A", "B", "C"],
            "name": ["Station A", "Station B", "Outside scope"],
            "lat": [41.9, 41.95, 42.0],
            "lng": [-87.6, -87.65, -87.7],
        }
    )
    return grid, weather, station_master


def training_features(grid, weather, station_master):
    frame = features.add_calendar_features(grid)
    frame = features.add_weather_features(frame, weather)
    frame = features.add_station_features(frame, station_master)
    return features.add_lag_features(frame, LAGS, ROLLS)


def test_train_serve_consistency():
    grid, weather, station_master = make_world()
    training_table = training_features(grid, weather, station_master)
    as_of = date(2025, 4, 10)
    history = grid.filter(pl.col("hour") <= datetime(2025, 4, 10, 23))

    inference_table = inference_features.build_inference_features(
        history,
        weather,
        station_master,
        as_of,
        LAGS,
        ROLLS,
    )
    expected = training_table.filter(
        pl.col("hour").dt.date() == date(2025, 4, 11)
    )
    actual = inference_table.sort("station_id", "hour")
    expected = expected.sort("station_id", "hour")
    assert actual.height == expected.height == 48

    for column in actual.columns:
        if column != "trips":
            assert actual[column].equals(expected[column]), column


def test_horizon_beyond_minimum_lag_is_rejected():
    grid, weather, station_master = make_world()
    with pytest.raises(ValueError, match="horizon"):
        inference_features.build_inference_features(
            grid,
            weather,
            station_master,
            date(2025, 4, 10),
            LAGS,
            ROLLS,
            horizon_hours=48,
        )


def test_missing_hour_for_one_station_is_rejected():
    grid, weather, station_master = make_world()
    broken = grid.filter(
        ~(
            (pl.col("station_id") == "B")
            & (pl.col("hour") == datetime(2025, 4, 10, 17))
        )
    )
    with pytest.raises(ValueError, match="incomplete"):
        inference_features.build_inference_features(
            broken,
            weather,
            station_master,
            date(2025, 4, 10),
            LAGS,
            ROLLS,
        )


def _small_config():
    return {
        "features": {"lag_hours": LAGS, "rolling_days": ROLLS},
        "model": {"features": MODEL_FEATURES, "params": PARAMS},
    }


def _train_and_save(tmp_path, train_day: date = date(2025, 4, 10)):
    grid, weather, station_master = make_world()
    table = training_features(grid, weather, station_master)
    cutoff = datetime(train_day.year, train_day.month, train_day.day, 23)
    training_slice = table.filter(pl.col("hour") <= cutoff)
    model, categories, val_mae = fit_lgbm(
        training_slice,
        MODEL_FEATURES,
        PARAMS,
        refit_full=True,
    )
    model_dir = tmp_path / f"model-{train_day}"
    save_artifact(
        model,
        categories,
        MODEL_FEATURES,
        PARAMS,
        ("2025-03-01", str(cutoff)),
        str(cutoff),
        val_mae,
        model_dir,
    )
    return grid, weather, station_master, model_dir


def test_artifact_roundtrip(tmp_path):
    grid, weather, station_master, model_dir = _train_and_save(tmp_path)
    model, metadata = load_artifact(model_dir)
    table = training_features(grid, weather, station_master).tail(200)
    x, _ = to_xy(table, MODEL_FEATURES, metadata["categories"])

    direct = model.predict(x)
    restored = predict_frame(model, metadata, table)
    assert np.allclose(np.maximum(direct, 0), restored.to_numpy())
    assert metadata["features"] == MODEL_FEATURES
    assert metadata["fitted_n_estimators"] == model.n_estimators_


def test_train_production_records_real_cutoff_and_refit(tmp_path):
    grid, weather, station_master = make_world()
    table = training_features(grid, weather, station_master)
    output = train_production(
        table,
        _small_config(),
        tmp_path / "production",
        date(2025, 4, 10),
    )
    model, metadata = load_artifact(output)
    assert metadata["train_through"] == "2025-04-10 23:00:00"
    assert metadata["fitted_n_estimators"] == model.n_estimators_


def test_batch_predict_is_idempotent_for_same_model(tmp_path):
    grid, weather, station_master, model_dir = _train_and_save(tmp_path)
    kwargs = {
        "history": grid,
        "weather": weather,
        "station_master": station_master,
        "cfg": _small_config(),
        "model_dir": model_dir,
        "out_dir": tmp_path / "predictions",
    }
    first = batch_predict.run_batch_predict(date(2025, 4, 10), **kwargs)
    first_mtime = first.stat().st_mtime_ns
    second = batch_predict.run_batch_predict(date(2025, 4, 10), **kwargs)
    output = pl.read_parquet(second)

    assert first == second
    assert second.stat().st_mtime_ns == first_mtime
    assert output.height == 48
    assert output["as_of"].unique().to_list() == [date(2025, 4, 10)]
    assert (output["prediction"] >= 0).all()


def test_batch_rejects_artifact_trained_after_replay_date(tmp_path):
    grid, weather, station_master, model_dir = _train_and_save(
        tmp_path,
        train_day=date(2025, 4, 11),
    )
    with pytest.raises(ValueError, match="leak"):
        batch_predict.run_batch_predict(
            date(2025, 4, 10),
            grid,
            weather,
            station_master,
            _small_config(),
            model_dir,
            tmp_path / "predictions",
        )


def test_existing_partition_from_different_model_is_rejected(tmp_path):
    grid, weather, station_master, model_dir = _train_and_save(tmp_path)
    output_dir = tmp_path / "predictions"
    batch_predict.run_batch_predict(
        date(2025, 4, 10),
        grid,
        weather,
        station_master,
        _small_config(),
        model_dir,
        output_dir,
    )
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model_version"] = "different-version"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="different-version|belongs"):
        batch_predict.run_batch_predict(
            date(2025, 4, 10),
            grid,
            weather,
            station_master,
            _small_config(),
            model_dir,
            output_dir,
        )


@pytest.fixture
def api_world(tmp_path):
    grid, weather, station_master, model_dir = _train_and_save(tmp_path)
    station_master_path = tmp_path / "station_master.parquet"
    station_master.write_parquet(station_master_path)
    predictions_dir = tmp_path / "predictions"
    batch_predict.run_batch_predict(
        date(2025, 4, 10),
        grid,
        weather,
        station_master,
        _small_config(),
        model_dir,
        predictions_dir,
    )
    client = TestClient(create_app(predictions_dir, station_master_path))
    return client, predictions_dir, grid


def test_api_health_and_forecast(api_world):
    client, _, _ = api_world
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["latest_as_of"] == "2025-04-10"
    assert health["model_version"]

    response = client.get("/forecast/A")
    assert response.status_code == 200
    assert len(response.json()["forecasts"]) == 24


def test_api_stations_only_returns_forecast_scope(api_world):
    client, _, _ = api_world
    station_ids = {row["station_id"] for row in client.get("/stations").json()}
    assert station_ids == {"A", "B"}


@pytest.mark.parametrize(
    ("path", "detail"),
    [
        ("/forecast/UNKNOWN", "scope"),
        ("/forecast/A?as_of=1999-01-01", "No forecast"),
    ],
)
def test_api_missing_resource_returns_404(api_world, path, detail):
    client, _, _ = api_world
    response = client.get(path)
    assert response.status_code == 404
    assert detail in response.json()["detail"]


def test_api_empty_store_is_valid(tmp_path):
    station_master = pl.DataFrame(
        {"station_id": ["A"], "name": ["Station A"]}
    )
    path = tmp_path / "station_master.parquet"
    station_master.write_parquet(path)
    client = TestClient(create_app(tmp_path / "empty", path))
    assert client.get("/health").json()["latest_as_of"] is None
    assert client.get("/stations").json() == []
    assert client.get("/forecast/A").status_code == 404


def test_evaluation_produces_week5_contract(api_world):
    _, predictions_dir, actuals = api_world
    detailed = prediction_actual_table(predictions_dir, actuals)
    report = evaluate_predictions(predictions_dir, actuals)
    assert {
        "as_of",
        "station_id",
        "hour",
        "prediction",
        "actual",
        "absolute_error",
    }.issubset(detailed.columns)
    assert report.height == 1
    assert report["n"][0] == 48
    assert report["mae"][0] >= 0
