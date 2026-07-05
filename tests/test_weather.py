from unittest import mock

import polars as pl
import pytest

from src.ingestion import weather

FAKE_RESPONSE = {
    "hourly": {
        "time": ["2024-05-01T00:00", "2024-05-01T01:00"],
        "temperature_2m": [12.3, 11.8],
        "precipitation": [0.0, 0.4],
        "snowfall": [0.0, 0.0],
        "wind_speed_10m": [8.1, 9.0],
        "relative_humidity_2m": [70, 72],
    }
}


def mocked_response():
    response = mock.Mock()
    response.json.return_value = FAKE_RESPONSE
    response.raise_for_status.return_value = None
    return response


def test_month_start():
    assert weather.month_start("2023-07") == "2023-07-01"


def test_month_end():
    assert weather.month_end("2026-02") == "2026-02-28"
    assert weather.month_end("2024-02") == "2024-02-29"
    assert weather.month_end("2026-06") == "2026-06-30"


def test_base_params_has_timezone():
    cfg = {
        "weather": {
            "latitude": 41.8781,
            "longitude": -87.6298,
            "timezone": "America/Chicago",
            "hourly_vars": ["temperature_2m", "precipitation"],
        }
    }

    params = weather.base_params(cfg)

    assert params["timezone"] == "America/Chicago"


def test_fetch_hourly_schema():
    with mock.patch.object(weather.requests, "get", return_value=mocked_response()):
        df = weather.fetch_hourly(weather.ARCHIVE_URL, {"fake": "params"})

    assert df.columns == weather.EXPECTED_WEATHER_COLUMNS
    assert df.schema["timestamp"] == pl.Datetime("us")
    assert df.height == 2


def test_validate_weather_schema_fails_loud():
    df = pl.DataFrame(
        {
            "timestamp": [],
            "temperature_2m": [],
        }
    )

    with pytest.raises(ValueError, match="Weather schema mismatch"):
        weather.validate_weather_schema(df, "bad-weather")
