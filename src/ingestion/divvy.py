"""Ingest Divvy trip data: list bucket -> download missing months -> validate -> parquet."""

import argparse
import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import polars as pl
import requests
import yaml

EXPECTED_COLUMNS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
]

SCHEMA_OVERRIDES = {
    "ride_id": pl.Utf8,
    "rideable_type": pl.Utf8,
    "started_at": pl.Utf8,
    "ended_at": pl.Utf8,
    "start_station_name": pl.Utf8,
    "start_station_id": pl.Utf8,
    "end_station_name": pl.Utf8,
    "end_station_id": pl.Utf8,
    "start_lat": pl.Float64,
    "start_lng": pl.Float64,
    "end_lat": pl.Float64,
    "end_lng": pl.Float64,
    "member_casual": pl.Utf8,
}

MONTH_FILE_RE = re.compile(r"^(\d{4})(\d{2})-divvy-tripdata\.zip$")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def list_bucket_keys(bucket_url: str) -> list[str]:
    """List all keys from a public S3 bucket using plain HTTP."""
    keys: list[str] = []
    token = None

    while True:
        params = {"list-type": "2"}
        if token:
            params["continuation-token"] = token

        resp = requests.get(bucket_url, params=params, timeout=30)
        resp.raise_for_status()

        root = ElementTree.fromstring(resp.content)
        keys += [el.text for el in root.findall(".//{*}Contents/{*}Key") if el.text]

        if root.findtext(".//{*}IsTruncated") != "true":
            return keys

        token = root.findtext(".//{*}NextContinuationToken")


def month_from_key(key: str) -> str | None:
    """Convert '202405-divvy-tripdata.zip' -> '2024-05'."""
    match = MONTH_FILE_RE.match(key)
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)}"


def validate_schema(df: pl.DataFrame, source_name: str) -> None:
    """Fail loudly if Divvy changes the CSV schema."""
    if df.columns != EXPECTED_COLUMNS:
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        extra = set(df.columns) - set(EXPECTED_COLUMNS)
        raise ValueError(
            f"Schema mismatch in {source_name}: "
            f"missing columns={sorted(missing)}, extra columns={sorted(extra)}"
        )


def read_trip_zip(zip_bytes: bytes, source_name: str) -> pl.DataFrame:
    """Read root CSV files from a Divvy zip and normalize timestamps."""
    frames = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [
            name for name in zf.namelist() if name.endswith(".csv") and "/" not in name
        ]

        if not csv_names:
            raise ValueError(f"No root CSV found in {source_name}")

        for name in csv_names:
            df = pl.read_csv(
                io.BytesIO(zf.read(name)),
                schema_overrides=SCHEMA_OVERRIDES,
                null_values=["", "NULL", "null"],
            )
            validate_schema(df, f"{source_name}/{name}")
            frames.append(df)

    df = pl.concat(frames)

    return df.with_columns(
        pl.col("started_at", "ended_at")
        .str.replace(r"\.\d+$", "")
        .str.to_datetime("%Y-%m-%d %H:%M:%S")
    )


def ingest(config_path: str) -> None:
    cfg = load_config(config_path)

    bucket_url = cfg["system"]["trip_bucket_url"]
    start_month = cfg["data"]["start_month"]
    end_month = cfg["data"]["end_month"]
    out_dir = Path(cfg["data"]["raw_dir"]) / "divvy"
    out_dir.mkdir(parents=True, exist_ok=True)

    available = {}
    for key in list_bucket_keys(bucket_url):
        month = month_from_key(key)
        if month and start_month <= month <= end_month:
            available[month] = key

    existing = {p.stem for p in out_dir.glob("*.parquet")}
    todo = sorted(set(available) - existing)

    print(
        f"Range {start_month}..{end_month}: "
        f"existing {len(existing)} months, downloading {len(todo)} months"
    )

    for month in todo:
        key = available[month]
        url = f"{bucket_url}/{key}"

        print(f"{month}: downloading {url}")
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()

        df = read_trip_zip(resp.content, key)
        out_path = out_dir / f"{month}.parquet"
        df.write_parquet(out_path)

        print(f"{month}: wrote {df.height:,} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    ingest(args.config)
