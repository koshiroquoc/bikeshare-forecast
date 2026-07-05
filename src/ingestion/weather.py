"""Ingest Open-Meteo weather data.

Two modes:
- historical: archive API for model training and EDA
- forecast: forecast API for future serving workflow

Both modes produce the same output schema:
timestamp + configured hourly weather variables.
"""

import argparse
from calendar import monthrange
from datetime import date
from pathlib import Path

import polars as pl
import requests
import yaml

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

EXPECTED_WEATHER_COLUMNS = [
    "timestamp",
    "temperature_2m",
    "precipitation",
    "snowfall",
    "wind_speed_10m",
    "relative_humidity_2m",
]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def month_start(month: str) -> str:
    """Convert '2023-07' -> '2023-07-01'."""
    return f"{month}-01"


def month_end(month: str) -> str:
    """Convert '2026-06' -> '2026-06-30'."""
    year, mon = month.split("-")
    year_i = int(year)
    mon_i = int(mon)
    last_day = monthrange(year_i, mon_i)[1]
    return f"{year}-{mon}-{last_day:02d}"


def base_params(cfg: dict) -> dict:
    weather_cfg = cfg["weather"]

    return {
        "latitude": weather_cfg["latitude"],
        "longitude": weather_cfg["longitude"],
        "hourly": ",".join(weather_cfg["hourly_vars"]),
        "timezone": weather_cfg["timezone"],
    }


def validate_weather_schema(df: pl.DataFrame, source_name: str) -> None:
    if df.columns != EXPECTED_WEATHER_COLUMNS:
        missing = set(EXPECTED_WEATHER_COLUMNS) - set(df.columns)
        extra = set(df.columns) - set(EXPECTED_WEATHER_COLUMNS)
        raise ValueError(
            f"Weather schema mismatch in {source_name}: "
            f"missing columns={sorted(missing)}, extra columns={sorted(extra)}"
        )


def fetch_hourly(url: str, params: dict) -> pl.DataFrame:
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()

    payload = resp.json()
    if "hourly" not in payload:
        raise ValueError(f"Open-Meteo response missing 'hourly': {payload}")

    df = (
        pl.DataFrame(payload["hourly"])
        .rename({"time": "timestamp"})
        .with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M"))
        .select(EXPECTED_WEATHER_COLUMNS)
    )

    validate_weather_schema(df, source_name=url)
    return df


def historical_output_path(cfg: dict) -> Path:
    start_date = month_start(cfg["data"]["start_month"])
    end_date = month_end(cfg["data"]["end_month"])

    return (
        Path(cfg["data"]["raw_dir"])
        / "weather"
        / "historical"
        / f"{start_date}_{end_date}.parquet"
    )


def forecast_output_path(cfg: dict) -> Path:
    return (
        Path(cfg["data"]["raw_dir"])
        / "weather"
        / "forecast"
        / f"{date.today().isoformat()}.parquet"
    )


def ingest_historical(cfg: dict, force: bool = False) -> Path:
    start_date = month_start(cfg["data"]["start_month"])
    end_date = month_end(cfg["data"]["end_month"])
    out_path = historical_output_path(cfg)

    if out_path.exists() and not force:
        print(f"Historical weather already exists, skipped: {out_path}")
        return out_path

    params = base_params(cfg) | {
        "start_date": start_date,
        "end_date": end_date,
    }

    df = fetch_hourly(ARCHIVE_URL, params)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)

    print(f"Historical weather: wrote {df.height:,} rows to {out_path}")
    return out_path


def ingest_forecast(cfg: dict, force: bool = False) -> Path:
    out_path = forecast_output_path(cfg)

    if out_path.exists() and not force:
        print(f"Forecast weather already exists, skipped: {out_path}")
        return out_path

    params = base_params(cfg) | {
        "forecast_days": 2,
    }

    df = fetch_hourly(FORECAST_URL, params)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)

    print(f"Forecast weather: wrote {df.height:,} rows to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--mode", choices=["historical", "forecast"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "historical":
        ingest_historical(cfg, force=args.force)
    else:
        ingest_forecast(cfg, force=args.force)


if __name__ == "__main__":
    main()
