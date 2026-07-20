"""Machine-readable performance and feature-drift monitoring gates."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from evidently import Report
from evidently.presets import DataDriftPreset

from src.serving.evaluate import evaluate_predictions, prediction_actual_table
from src.serving.model_artifact import load_metadata

DRIFT_COLUMNS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "lag_24",
    "lag_168",
    "roll_mean_7d",
]


def performance_gate(
    daily_mae: pl.DataFrame,
    baseline_mae: float,
    tolerance: float = 0.20,
) -> dict:
    """Fail when recent serving MAE exceeds the artifact promise plus tolerance."""
    if daily_mae.is_empty():
        raise ValueError("Performance gate needs at least one evaluated day.")
    if baseline_mae <= 0:
        raise ValueError("baseline_mae must be positive.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    mean_mae = float(daily_mae["mae"].mean())
    threshold = baseline_mae * (1 + tolerance)
    return {
        "gate": "performance",
        "passed": mean_mae <= threshold,
        "mean_mae": round(mean_mae, 4),
        "baseline_mae": round(float(baseline_mae), 4),
        "threshold": round(threshold, 4),
        "tolerance": tolerance,
        "n_days": daily_mae.height,
    }


def _walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _drift_share(snapshot) -> float:
    """Extract dataset drift share while keeping Evidently API changes loud."""
    payload = snapshot.dict()
    for node in _walk(payload):
        if not isinstance(node, dict):
            continue
        metric_name = str(
            node.get("metric_name")
            or node.get("metric_id")
            or node.get("type")
            or ""
        )
        if "DriftedColumnsCount" not in metric_name:
            continue
        for value_node in _walk(node.get("value", node)):
            if isinstance(value_node, dict) and "share" in value_node:
                return float(value_node["share"])
    raise RuntimeError(
        "Could not find DriftedColumnsCount.share in the Evidently report. "
        "The pinned Evidently API may have changed."
    )


def drift_gate(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    out_html: str | Path | None = None,
    max_drift_share: float = 0.50,
) -> dict:
    """Compare current model inputs with a recent pre-serving reference window."""
    if not 0 <= max_drift_share <= 1:
        raise ValueError("max_drift_share must be between 0 and 1.")

    columns = [
        column
        for column in DRIFT_COLUMNS
        if column in reference.columns and column in current.columns
    ]
    columns = [
        column
        for column in columns
        if reference[column].std() is not None
        and reference[column].std() > 1e-9
    ]
    if not columns:
        return {
            "gate": "drift",
            "passed": True,
            "drift_share": 0.0,
            "max_drift_share": max_drift_share,
            "n_columns": 0,
            "columns": [],
            "report_html": None,
            "note": "No shared non-constant columns were available.",
        }

    reference_pd = reference.select(columns).drop_nulls().to_pandas()
    current_pd = current.select(columns).drop_nulls().to_pandas()
    if reference_pd.empty or current_pd.empty:
        raise ValueError("Drift gate received no complete rows after null filtering.")

    snapshot = Report(
        [DataDriftPreset(drift_share=max_drift_share)]
    ).run(
        current_data=current_pd,
        reference_data=reference_pd,
    )
    if out_html is not None:
        out_html = Path(out_html)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(out_html))

    share = _drift_share(snapshot)
    return {
        "gate": "drift",
        "passed": share <= max_drift_share,
        "drift_share": round(share, 3),
        "max_drift_share": max_drift_share,
        "n_columns": len(columns),
        "columns": columns,
        "reference_rows": len(reference_pd),
        "current_rows": len(current_pd),
        "report_html": str(out_html) if out_html else None,
    }


def run_monitoring(
    features_df: pl.DataFrame,
    predictions_dir: str | Path,
    model_dir: str | Path,
    *,
    baseline_mae: float | None = None,
    performance_tolerance: float = 0.20,
    max_drift_share: float = 0.50,
    reference_days: int = 60,
    report_dir: str | Path = "reports",
) -> dict:
    """Run both gates against the exact artifact version that made predictions."""
    if reference_days < 1:
        raise ValueError("reference_days must be at least 1.")

    metadata = load_metadata(model_dir)
    actuals = features_df.select("station_id", "hour", "trips")
    detailed = prediction_actual_table(predictions_dir, actuals)
    versions = sorted(detailed["model_version"].unique().to_list())
    expected_version = metadata["model_version"]
    if versions != [expected_version]:
        raise ValueError(
            f"Prediction store model versions {versions} do not match "
            f"artifact version {expected_version}."
        )

    daily = evaluate_predictions(predictions_dir, actuals)
    promised_mae = (
        float(baseline_mae)
        if baseline_mae is not None
        else float(metadata["val_mae"])
    )
    performance = performance_gate(
        daily,
        promised_mae,
        performance_tolerance,
    )

    served_dates = daily["target_date"].unique().sort().to_list()
    current = features_df.filter(
        pl.col("hour").dt.date().is_in(served_dates)
    )
    train_through = datetime.fromisoformat(metadata["train_through"])
    first_served = datetime.combine(served_dates[0], datetime.min.time())
    reference_end = min(
        train_through,
        first_served - timedelta(hours=1),
    )
    reference_start = reference_end - timedelta(
        hours=reference_days * 24 - 1
    )
    reference = features_df.filter(
        pl.col("hour").is_between(
            reference_start,
            reference_end,
            closed="both",
        )
    )

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    drift = drift_gate(
        reference,
        current,
        report_dir / "drift_report.html",
        max_drift_share,
    )

    if not performance["passed"]:
        severity = "critical"
    elif not drift["passed"]:
        severity = "warning"
    else:
        severity = "ok"

    summary = {
        "status": severity,
        "passed": severity == "ok",
        "model_version": expected_version,
        "model_dir": str(model_dir),
        "predictions_dir": str(predictions_dir),
        "reference_window": [
            str(reference_start),
            str(reference_end),
        ],
        "served_dates": [str(day) for day in served_dates],
        "gates": [performance, drift],
    }
    (report_dir / "monitoring_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        f"[monitoring] {severity.upper()} | "
        f"performance={performance['mean_mae']}"
        f"/{performance['threshold']} | "
        f"drift={drift['drift_share']}"
        f"/{drift['max_drift_share']}"
    )
    return summary
