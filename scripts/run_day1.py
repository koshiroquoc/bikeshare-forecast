"""Run Week 3 Day 1 baseline + default LightGBM experiment through MLflow."""

import polars as pl
import yaml

from src.training import backtest
from src.training.models import LGBMPredictor
from src.training.run_experiment import run_experiment


def main() -> None:
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    df = pl.read_parquet("data/processed/features.parquet")

    predictors = [
        backtest.SeasonalNaive(),
        backtest.HistoricalMean(),
        LGBMPredictor(
            cfg["model"]["features"],
            cfg["model"]["params"],
            name="lgbm",
        ),
    ]

    summary, run_id = run_experiment(
        features_df=df,
        predictors=predictors,
        n_windows=cfg["backtest"]["n_windows"],
        run_name="day1-baselines-and-default-lgbm",
        description="Week 3 Day 1: seasonal naive, historical mean, and default LightGBM.",
        config=cfg,
    )

    print(f"MLflow run ID: {run_id}")
    print(summary)


if __name__ == "__main__":
    main()
