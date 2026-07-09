"""Build processed station-hour feature table.

Pipeline:
raw trips -> cleaning -> station scope -> station-hour grid -> features.parquet
"""

import argparse
from pathlib import Path

import polars as pl
import yaml

from src.processing import aggregate, cleaning, features


def load_config(path: str) -> dict:
    """Load YAML project config."""
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str) -> None:
    """Build processed feature table and supporting data products."""
    cfg = load_config(config_path)

    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    trips = pl.scan_parquet(str(raw_dir / "divvy" / "*.parquet"))
    mapping = pl.read_csv("data/reference/station_mapping.csv")
    weather = pl.read_parquet(str(raw_dir / "weather" / "historical" / "*.parquet"))

    report = cleaning.removal_report(trips, cfg)
    total = report.pop("total")

    print(f"Total raw trips: {total:,}")
    print("Rows violating each cleaning rule:")
    for rule, n_rows in report.items():
        print(f"  {rule:<24} {n_rows:>12,} ({n_rows / total:.2%})")

    clean = cleaning.clean_trips(trips, cfg, mapping)

    station_master = aggregate.build_station_master(clean)
    station_month_panel = aggregate.build_station_month_panel(clean)

    station_master.write_parquet(processed_dir / "station_master.parquet")
    station_month_panel.write_parquet(processed_dir / "station_month_panel.parquet")

    scope = aggregate.select_scope(clean, cfg["scope"]["top_n_stations"])
    counts = aggregate.aggregate_station_hour(clean, scope)
    grid = aggregate.build_full_grid(counts, scope)

    n_hours = grid["hour"].n_unique()
    expected_rows = len(scope) * n_hours

    print(
        f"Full grid: {grid.height:,} rows = {len(scope):,} stations x {n_hours:,} hours"
    )

    if grid.height != expected_rows:
        raise ValueError(
            f"Full grid row count mismatch: expected {expected_rows:,}, "
            f"got {grid.height:,}"
        )

    grid = aggregate.trim_to_station_lifetime(grid, clean)
    print(f"After station lifetime trim: {grid.height:,} rows")

    df = features.add_calendar_features(grid)
    df = features.add_weather_features(df, weather)
    df = features.add_station_features(df, station_master)
    df = features.add_lag_features(
        df,
        cfg["features"]["lag_hours"],
        cfg["features"]["rolling_days"],
    )

    output_path = processed_dir / "features.parquet"
    df.write_parquet(output_path)

    print(f"{output_path}: {df.height:,} rows x {df.width:,} columns")
    print(f"Total trips in features: {df['trips'].sum():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    main(args.config)
