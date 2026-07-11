"""Run Week 3 Day 4 feature-iteration experiments.

Variants:
1. LGBM + station demand-shape cluster
2. LGBM + major Chicago event flag
3. LGBM without weather features

Each variant uses the same rolling-origin backtest framework and MLflow logging.
"""

import polars as pl
import yaml

from src.training import backtest, clustering, events
from src.training.models import LGBMPredictor
from src.training.run_experiment import run_experiment

WEATHER_FEATURES = {
    "temperature_2m",
    "precipitation",
    "snowfall",
    "wind_speed_10m",
    "relative_humidity_2m",
}


def load_config() -> dict:
    """Load project config."""
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def print_summary(title: str, summary: pl.DataFrame, run_id: str) -> None:
    """Print experiment summary consistently."""
    print("\n" + "=" * 80)
    print(title)
    print(f"MLflow run ID: {run_id}")
    print(summary)
    print("=" * 80 + "\n")


def event_coverage(df: pl.DataFrame, event_days: pl.DataFrame) -> pl.DataFrame:
    """Show how many event station-hours exist in each backtest window."""
    df_event = events.add_event_feature(
        df.select(["station_id", "hour", "trips"]), event_days
    )

    return (
        df_event.with_columns(pl.col("hour").dt.strftime("%Y-%m").alias("_eval_month"))
        .group_by("_eval_month")
        .agg(
            pl.col("is_major_event").sum().alias("event_station_hours"),
            pl.len().alias("station_hours"),
        )
        .with_columns(
            (pl.col("event_station_hours") / pl.col("station_hours")).alias(
                "event_station_hour_share"
            )
        )
        .sort("_eval_month")
    )


def run_cluster_variant(df: pl.DataFrame, cfg: dict) -> tuple[pl.DataFrame, str]:
    """Run LGBM with station demand-shape cluster."""
    first_eval_month = backtest.eval_months(df, cfg["backtest"]["n_windows"])[0]

    clustering_base = df.filter(
        pl.col("hour").dt.strftime("%Y-%m") < first_eval_month
    ).select(["station_id", "hour", "trips"])

    clusters = clustering.cluster_stations(
        clustering_base,
        k=5,
        seed=42,
    )

    df_clustered = clustering.add_cluster_feature(df, clusters)
    cluster_features = [*cfg["model"]["features"], "cluster"]

    predictors = [
        backtest.SeasonalNaive(),
        backtest.HistoricalMean(),
        LGBMPredictor(
            cluster_features,
            cfg["model"]["params"],
            name="lgbm_cluster",
        ),
    ]

    summary, run_id = run_experiment(
        features_df=df_clustered,
        predictors=predictors,
        n_windows=cfg["backtest"]["n_windows"],
        run_name="day4-lgbm-cluster",
        description=(
            "Week 3 Day 4: LightGBM with station demand-shape cluster feature. "
            "Clusters are computed only from data before the first evaluation window."
        ),
        config=cfg,
    )

    clusters.write_parquet("data/processed/station_clusters_day4.parquet")

    return summary, run_id


def run_event_variant(df: pl.DataFrame, cfg: dict) -> tuple[pl.DataFrame, str]:
    """Run LGBM with major-event flag."""
    event_days = events.load_event_days("data/reference/chicago_events.csv")

    print("\nEvent-day coverage by month:")
    print(event_coverage(df, event_days).tail(12))

    df_event = events.add_event_feature(df, event_days)
    event_features = [*cfg["model"]["features"], "is_major_event"]

    predictors = [
        backtest.SeasonalNaive(),
        backtest.HistoricalMean(),
        LGBMPredictor(
            event_features,
            cfg["model"]["params"],
            name="lgbm_event",
        ),
    ]

    summary, run_id = run_experiment(
        features_df=df_event,
        predictors=predictors,
        n_windows=cfg["backtest"]["n_windows"],
        run_name="day4-lgbm-event",
        description=(
            "Week 3 Day 4: LightGBM with manually curated Chicago major-event flag."
        ),
        config=cfg,
    )

    return summary, run_id


def run_no_weather_variant(df: pl.DataFrame, cfg: dict) -> tuple[pl.DataFrame, str]:
    """Run LGBM without weather features."""
    no_weather_features = [
        feature
        for feature in cfg["model"]["features"]
        if feature not in WEATHER_FEATURES
    ]

    predictors = [
        backtest.SeasonalNaive(),
        backtest.HistoricalMean(),
        LGBMPredictor(
            no_weather_features,
            cfg["model"]["params"],
            name="lgbm_no_weather",
        ),
    ]

    summary, run_id = run_experiment(
        features_df=df,
        predictors=predictors,
        n_windows=cfg["backtest"]["n_windows"],
        run_name="day4-lgbm-no-weather",
        description=(
            "Week 3 Day 4: LightGBM without weather features. "
            "This quantifies the contribution of weather features and the risk of "
            "forecast-weather train/serve skew."
        ),
        config=cfg,
    )

    return summary, run_id


def main() -> None:
    cfg = load_config()
    df = pl.read_parquet("data/processed/features.parquet")

    print("Feature table:", df.shape)
    print("Backtest windows:", backtest.eval_months(df, cfg["backtest"]["n_windows"]))

    cluster_summary, cluster_run_id = run_cluster_variant(df, cfg)
    print_summary("Cluster variant", cluster_summary, cluster_run_id)

    event_summary, event_run_id = run_event_variant(df, cfg)
    print_summary("Event variant", event_summary, event_run_id)

    no_weather_summary, no_weather_run_id = run_no_weather_variant(df, cfg)
    print_summary("No-weather variant", no_weather_summary, no_weather_run_id)


if __name__ == "__main__":
    main()
