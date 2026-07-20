"""Leakage-safe champion/challenger promotion on a truly unseen full month."""

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml

from src.serving.model_artifact import (
    load_artifact,
    load_metadata,
    predict_frame,
)
from src.training import backtest
from src.training.models import LGBMPredictor
from src.training.train_production import train_production


class ArtifactPredictor:
    """Expose a frozen production artifact through the predictor interface."""

    name = "current"

    def __init__(self, model_dir: str | Path):
        self.model, self.metadata = load_artifact(model_dir)

    def predict(
        self,
        train_df: pl.DataFrame,
        eval_df: pl.DataFrame,
    ) -> pl.Series:
        del train_df
        return predict_frame(self.model, self.metadata, eval_df)


def latest_unseen_full_month(
    features_df: pl.DataFrame,
    train_through: datetime,
) -> str | None:
    """Return the latest full month starting strictly after champion training."""
    month_count = features_df["hour"].dt.strftime("%Y-%m").n_unique()
    full_months = backtest.eval_months(features_df, month_count)
    eligible = [
        month
        for month in full_months
        if datetime.strptime(month, "%Y-%m") > train_through
    ]
    return eligible[-1] if eligible else None


def _promote_atomically(
    features_df: pl.DataFrame,
    cfg: dict,
    current_dir: Path,
    archive_dir: Path,
    current_version: str,
) -> tuple[Path, dict]:
    """Train in staging, archive champion, then atomically swap directories."""
    current_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".candidate-",
            dir=current_dir.parent,
        )
    )
    train_production(features_df, cfg, str(staging_dir))
    new_metadata = load_metadata(staging_dir)

    archive_path = archive_dir / current_version
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        raise FileExistsError(
            f"Archive already exists for model version {current_version}: "
            f"{archive_path}"
        )

    current_dir.rename(archive_path)
    try:
        staging_dir.rename(current_dir)
    except Exception:
        archive_path.rename(current_dir)
        raise
    return archive_path, new_metadata


def promote_if_better(
    features_df: pl.DataFrame,
    cfg: dict,
    current_dir: str | Path = "models/current",
    archive_dir: str | Path = "models/archive",
    min_improvement: float = 0.02,
) -> dict:
    """Promote only after a fair comparison on a month neither model has seen."""
    if not 0 <= min_improvement < 1:
        raise ValueError("min_improvement must be between 0 and 1.")

    current_dir = Path(current_dir)
    archive_dir = Path(archive_dir)
    current = ArtifactPredictor(current_dir)
    train_through = datetime.fromisoformat(
        current.metadata["train_through"]
    )
    eval_month = latest_unseen_full_month(features_df, train_through)
    if eval_month is None:
        decision = {
            "status": "waiting_for_unseen_month",
            "promoted": False,
            "current_version": current.metadata["model_version"],
            "current_train_through": str(train_through),
            "eval_month": None,
            "reason": (
                "No complete calendar month starts after the current artifact "
                "training cutoff. Promotion evaluation would leak."
            ),
        }
        print("[promote] WAIT | no unseen full month is available")
        return decision

    candidate = LGBMPredictor(
        cfg["model"]["features"],
        cfg["model"]["params"],
        name="candidate",
    )
    predictions = backtest.prediction_table(
        features_df,
        [candidate, current],
        eval_month,
    )
    mae_candidate = float(
        backtest.mae(
            predictions["trips"],
            predictions["pred_candidate"],
        )
    )
    mae_current = float(
        backtest.mae(
            predictions["trips"],
            predictions["pred_current"],
        )
    )
    required_mae = mae_current * (1 - min_improvement)
    should_promote = mae_candidate <= required_mae
    decision = {
        "status": "promoted" if should_promote else "kept",
        "promoted": should_promote,
        "mae_candidate": round(mae_candidate, 4),
        "mae_current": round(mae_current, 4),
        "required_candidate_mae": round(required_mae, 4),
        "min_improvement": min_improvement,
        "eval_month": eval_month,
        "current_version": current.metadata["model_version"],
        "current_train_through": str(train_through),
        "note": (
            "Both champion and challenger were evaluated on a complete month "
            "starting after the champion cutoff."
        ),
    }

    if should_promote:
        archive_path, new_metadata = _promote_atomically(
            features_df,
            cfg,
            current_dir,
            archive_dir,
            current.metadata["model_version"],
        )
        decision["archived_to"] = str(archive_path)
        decision["new_version"] = new_metadata["model_version"]

    print(
        f"[promote] {decision['status'].upper()} | "
        f"candidate={decision['mae_candidate']} | "
        f"current={decision['mae_current']} | "
        f"required<={decision['required_candidate_mae']} | "
        f"eval={eval_month}"
    )
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--current-dir", default="models/current")
    parser.add_argument("--archive-dir", default="models/archive")
    parser.add_argument("--min-improvement", type=float, default=0.02)
    parser.add_argument(
        "--decision-out",
        default="reports/promotion_decision.json",
    )
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    features = pl.read_parquet("data/processed/features.parquet")
    decision = promote_if_better(
        features,
        config,
        current_dir=args.current_dir,
        archive_dir=args.archive_dir,
        min_improvement=args.min_improvement,
    )
    output_path = Path(args.decision_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(decision, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))
