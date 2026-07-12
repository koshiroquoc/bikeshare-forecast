"""Run Week 3 Day 5 LightGBM tuning experiments.

This is intentionally a small, disciplined search. We tune only the highest
impact LightGBM knobs and evaluate every candidate through the same rolling-origin
backtest harness used by the baselines and default model.

The sweep uses a lower n_estimators cap for speed. The selected final candidate
should be rerun separately with n_estimators=2000.
"""

import json
from typing import Any

import polars as pl
import yaml

from src.training import backtest
from src.training.models import LGBMPredictor
from src.training.run_experiment import run_experiment

SWEEP_N_ESTIMATORS = 500


def load_config() -> dict[str, Any]:
    """Load project config."""
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def base_sweep_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return default model params with a faster estimator cap for tuning."""
    params = dict(cfg["model"]["params"])
    params["n_estimators"] = SWEEP_N_ESTIMATORS
    return params


def candidate_specs() -> list[tuple[str, dict[str, Any]]]:
    """Small one-knob-at-a-time search around the default LightGBM config."""
    return [
        ("lr_0_03", {"learning_rate": 0.03}),
        ("lr_0_05", {"learning_rate": 0.05}),
        ("lr_0_10", {"learning_rate": 0.10}),
        ("leaves_31", {"num_leaves": 31}),
        ("leaves_63", {"num_leaves": 63}),
        ("leaves_127", {"num_leaves": 127}),
        ("min_child_20", {"min_child_samples": 20}),
        ("min_child_100", {"min_child_samples": 100}),
        ("min_child_500", {"min_child_samples": 500}),
    ]


def lgbm_row(summary: pl.DataFrame) -> dict[str, Any]:
    """Extract the LGBM row from a run summary."""
    rows = summary.filter(pl.col("predictor").str.starts_with("lgbm_tune_")).to_dicts()

    if len(rows) != 1:
        raise ValueError(f"Expected exactly one tuning LGBM row, found {len(rows)}.")

    return rows[0]


def run_candidate(
    df: pl.DataFrame,
    cfg: dict[str, Any],
    name: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Run one tuning candidate and return one leaderboard row."""
    params = base_sweep_params(cfg) | overrides
    predictor_name = f"lgbm_tune_{name}"

    predictors = [
        backtest.SeasonalNaive(),
        backtest.HistoricalMean(),
        LGBMPredictor(
            cfg["model"]["features"],
            params,
            name=predictor_name,
        ),
    ]

    run_config = dict(cfg)
    run_config["day5_tuning_candidate"] = {
        "name": name,
        "params": params,
    }

    summary, run_id = run_experiment(
        features_df=df,
        predictors=predictors,
        n_windows=cfg["backtest"]["n_windows"],
        run_name=f"day5-tune-{name}",
        description=(
            "Week 3 Day 5: small LightGBM tuning sweep. "
            f"Candidate={name}, overrides={overrides}."
        ),
        config=run_config,
    )

    row = lgbm_row(summary)

    return {
        "candidate": name,
        "run_id": run_id,
        "params": json.dumps(params, sort_keys=True),
        "mae_mean": row["mae_mean"],
        "mae_std": row["mae_std"],
        "rmse_mean": row["rmse_mean"],
        "mase_mean": row["mase_mean"],
    }


def main() -> None:
    cfg = load_config()
    df = pl.read_parquet("data/processed/features.parquet")

    print("Feature table:", df.shape)
    print("Backtest windows:", backtest.eval_months(df, cfg["backtest"]["n_windows"]))
    print("Sweep n_estimators:", SWEEP_N_ESTIMATORS)

    rows = []

    for name, overrides in candidate_specs():
        print("\n" + "=" * 80)
        print(f"Running candidate: {name}")
        print("Overrides:", overrides)

        row = run_candidate(df, cfg, name, overrides)
        rows.append(row)

        print("Run ID:", row["run_id"])
        print(
            "MAE:",
            round(row["mae_mean"], 4),
            "±",
            round(row["mae_std"], 4),
            "RMSE:",
            round(row["rmse_mean"], 4),
            "MASE:",
            round(row["mase_mean"], 4),
        )

    leaderboard = pl.DataFrame(rows).sort("mae_mean")

    out_path = "data/processed/day5_tuning_leaderboard.csv"
    leaderboard.write_csv(out_path)

    print("\n" + "=" * 80)
    print("Day 5 tuning leaderboard")
    print(leaderboard)
    print(f"\nSaved leaderboard to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
