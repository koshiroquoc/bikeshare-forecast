"""MLflow experiment runner for rolling-origin backtests.

Rule: no model result should go into README without a corresponding MLflow run ID.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import polars as pl
import yaml

from src.training import backtest

EXPERIMENT_NAME = "bikeshare-forecast"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def git_hash() -> str:
    """Return the current short Git commit hash, or 'unknown' outside Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _safe_log_metric(name: str, value: Any) -> None:
    """Log metric only when MLflow can accept the value."""
    if value is None:
        return
    mlflow.log_metric(name, float(value))


def _log_predictor_params(predictors: list) -> None:
    """Log feature lists and model params for each predictor."""
    mlflow.log_param("predictors", ",".join(p.name for p in predictors))

    for predictor in predictors:
        if hasattr(predictor, "features"):
            features = list(predictor.features)
            mlflow.log_param(f"{predictor.name}_n_features", len(features))
            mlflow.log_param(f"{predictor.name}_features", json.dumps(features))

        if hasattr(predictor, "params"):
            for key, value in predictor.params.items():
                mlflow.log_param(f"{predictor.name}_{key}", value)


def _write_artifacts(
    tmp_dir: Path,
    results: pl.DataFrame,
    summary: pl.DataFrame,
    predictors: list,
    config: dict | None,
) -> None:
    """Write run artifacts before logging the artifact directory to MLflow."""
    results.write_csv(tmp_dir / "results_per_window.csv")
    summary.write_csv(tmp_dir / "summary.csv")

    if config is not None:
        with open(tmp_dir / "config_snapshot.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    for predictor in predictors:
        model = getattr(predictor, "last_model_", None)
        if model is None:
            continue

        importance = pl.DataFrame(
            {
                "feature": model.feature_name_,
                "gain": model.booster_.feature_importance(importance_type="gain"),
            }
        ).sort("gain", descending=True)

        importance.write_csv(tmp_dir / f"importance_{predictor.name}.csv")


def run_experiment(
    features_df: pl.DataFrame,
    predictors: list,
    n_windows: int,
    run_name: str,
    description: str = "",
    config: dict | None = None,
) -> tuple[pl.DataFrame, str]:
    """Run backtest, log params/metrics/artifacts to MLflow, and return summary + run ID."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = backtest.run_backtest(features_df, predictors, n_windows)
    summary = backtest.summarize(results)

    with mlflow.start_run(run_name=run_name, description=description) as run:
        run_id = run.info.run_id

        mlflow.log_param("git_commit", git_hash())
        mlflow.log_param("n_windows", n_windows)
        mlflow.log_param("n_rows", features_df.height)
        mlflow.log_param("n_cols", features_df.width)

        _log_predictor_params(predictors)

        for row in results.iter_rows(named=True):
            predictor = row["predictor"]
            window = row["window"]

            _safe_log_metric(f"mae_{predictor}_{window}", row["mae"])
            _safe_log_metric(f"rmse_{predictor}_{window}", row["rmse"])

        for row in summary.iter_rows(named=True):
            predictor = row["predictor"]

            _safe_log_metric(f"mae_mean_{predictor}", row["mae_mean"])
            _safe_log_metric(f"mae_std_{predictor}", row["mae_std"])
            _safe_log_metric(f"rmse_mean_{predictor}", row["rmse_mean"])
            _safe_log_metric(f"mase_mean_{predictor}", row["mase_mean"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _write_artifacts(tmp_dir, results, summary, predictors, config)
            mlflow.log_artifacts(str(tmp_dir))

    return summary, run_id
