"""Prefect orchestration for one historical replay-style forecast day."""

import argparse
from datetime import date

import polars as pl
import yaml
from prefect import flow, get_run_logger, task

from src.serving.batch_predict import run_batch_predict


@task(retries=2, retry_delay_seconds=10)
def load_history() -> pl.DataFrame:
    return pl.read_parquet(
        "data/processed/features.parquet",
        columns=["station_id", "hour", "trips"],
    )


@task(retries=2, retry_delay_seconds=30)
def load_historical_weather() -> pl.DataFrame:
    return pl.read_parquet("data/raw/weather/historical/*.parquet")


@task
def load_station_master() -> pl.DataFrame:
    return pl.read_parquet("data/processed/station_master.parquet")


@flow(name="nightly-forecast", log_prints=True)
def nightly_forecast(
    as_of: str,
    model_dir: str = "models/current",
    out_dir: str = "data/predictions",
) -> str:
    """Run the production path with historical weather for honest replay."""
    logger = get_run_logger()
    with open("config/config.yaml") as config_file:
        config = yaml.safe_load(config_file)

    output = run_batch_predict(
        as_of=date.fromisoformat(as_of),
        history=load_history(),
        weather=load_historical_weather(),
        station_master=load_station_master(),
        cfg=config,
        model_dir=model_dir,
        out_dir=out_dir,
    )
    logger.info("Forecast completed: %s", output)
    return str(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--model-dir", default="models/current")
    parser.add_argument("--out-dir", default="data/predictions")
    args = parser.parse_args()
    nightly_forecast(
        as_of=args.as_of,
        model_dir=args.model_dir,
        out_dir=args.out_dir,
    )
