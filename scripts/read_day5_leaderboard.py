"""Read Day 5 tuning leaderboard from already-logged MLflow runs."""

import json
import os

import mlflow
import polars as pl

EXPERIMENT_NAME = "bikeshare-forecast"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def _metric(metrics: dict[str, float], predictor: str, metric: str) -> float | None:
    """Read a metric from the run's metric dictionary."""
    return metrics.get(f"{metric}_{predictor}")


def _params_for_candidate(params: dict[str, str], predictor: str) -> str:
    """Extract only LightGBM hyperparameters for one tuning predictor."""
    allowed_keys = {
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "min_child_samples",
    }

    prefix = f"{predictor}_"

    extracted = {
        key.removeprefix(prefix): value
        for key, value in params.items()
        if key.startswith(prefix) and key.removeprefix(prefix) in allowed_keys
    }

    return json.dumps(extracted, sort_keys=True)


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)

    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if exp is None:
        raise ValueError(f"Experiment not found: {EXPERIMENT_NAME}")

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.run_name LIKE 'day5-tune-%'",
        output_format="list",
    )

    rows = []

    for run in runs:
        run_name = run.info.run_name
        candidate = run_name.removeprefix("day5-tune-")
        predictor = f"lgbm_tune_{candidate}"

        rows.append(
            {
                "candidate": candidate,
                "run_id": run.info.run_id,
                "params": _params_for_candidate(run.data.params, predictor),
                "mae_mean": _metric(run.data.metrics, predictor, "mae_mean"),
                "mae_std": _metric(run.data.metrics, predictor, "mae_std"),
                "rmse_mean": _metric(run.data.metrics, predictor, "rmse_mean"),
                "mase_mean": _metric(run.data.metrics, predictor, "mase_mean"),
            }
        )

    leaderboard = pl.DataFrame(rows).drop_nulls(subset=["mae_mean"]).sort("mae_mean")

    out_path = "data/processed/day5_tuning_leaderboard.csv"
    leaderboard.write_csv(out_path)

    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Runs found:", len(runs))
    print(leaderboard)
    print(f"\nSaved leaderboard to {out_path}")


if __name__ == "__main__":
    main()
