"""FastAPI application that reads precomputed forecasts, never the model."""

import re
from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

AS_OF_RE = re.compile(r"as_of=(\d{4}-\d{2}-\d{2})\.parquet$")


class HourlyForecast(BaseModel):
    hour: str
    prediction: float


class ForecastResponse(BaseModel):
    station_id: str
    as_of: str
    model_version: str
    forecasts: list[HourlyForecast]


def create_app(
    predictions_dir: str | Path = "data/predictions",
    station_master_path: str | Path = "data/processed/station_master.parquet",
) -> FastAPI:
    """Application factory: tests can inject temporary stores without patching."""
    store = Path(predictions_dir)
    app = FastAPI(title="Divvy Demand Forecast", version="1.0")

    def partitions() -> dict[str, Path]:
        found = {}
        for path in sorted(store.glob("as_of=*.parquet")):
            match = AS_OF_RE.search(path.name)
            if match:
                found[match.group(1)] = path
        return found

    @app.get("/health")
    def health():
        available = partitions()
        latest = max(available) if available else None
        model_version = None
        if latest:
            model_version = pl.read_parquet(
                available[latest],
                columns=["model_version"],
            )["model_version"][0]
        return {
            "status": "ok",
            "n_days_available": len(available),
            "latest_as_of": latest,
            "model_version": model_version,
        }

    @app.get("/stations")
    def stations():
        available = partitions()
        if not available:
            return []
        latest_path = available[max(available)]
        scope = pl.read_parquet(
            latest_path,
            columns=["station_id"],
        ).unique()
        station_master = pl.read_parquet(
            station_master_path,
            columns=["station_id", "name"],
        )
        return station_master.join(scope, on="station_id", how="inner").to_dicts()

    @app.get(
        "/forecast/{station_id}",
        response_model=ForecastResponse,
    )
    def forecast(station_id: str, as_of: str | None = None):
        available = partitions()
        if not available:
            raise HTTPException(
                404,
                "No forecasts are available. Run the forecast flow first.",
            )
        selected_date = as_of or max(available)
        if selected_date not in available:
            raise HTTPException(
                404,
                f"No forecast for as_of={selected_date}. "
                f"Recent available dates: {sorted(available)[-5:]}",
            )

        frame = pl.read_parquet(available[selected_date]).filter(
            pl.col("station_id") == station_id
        )
        if frame.is_empty():
            raise HTTPException(
                404,
                f"Station '{station_id}' is outside the forecast scope.",
            )
        return ForecastResponse(
            station_id=station_id,
            as_of=selected_date,
            model_version=frame["model_version"][0],
            forecasts=[
                HourlyForecast(
                    hour=str(row["hour"]),
                    prediction=round(row["prediction"], 2),
                )
                for row in frame.sort("hour").iter_rows(named=True)
            ],
        )

    return app


app = create_app()
