"""Run monitoring and expose severity through process exit codes."""

import argparse
import json
import sys

import polars as pl

from src.monitoring.checks import run_monitoring


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--predictions-dir", default="data/predictions")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--baseline-mae", type=float)
    parser.add_argument("--performance-tolerance", type=float, default=0.20)
    parser.add_argument("--max-drift-share", type=float, default=0.50)
    parser.add_argument("--reference-days", type=int, default=60)
    args = parser.parse_args()

    features = pl.read_parquet("data/processed/features.parquet")
    summary = run_monitoring(
        features,
        predictions_dir=args.predictions_dir,
        model_dir=args.model_dir,
        baseline_mae=args.baseline_mae,
        performance_tolerance=args.performance_tolerance,
        max_drift_share=args.max_drift_share,
        reference_days=args.reference_days,
        report_dir=args.report_dir,
    )
    print(json.dumps(summary, indent=2))
    exit_codes = {"ok": 0, "warning": 1, "critical": 2}
    sys.exit(exit_codes[summary["status"]])
