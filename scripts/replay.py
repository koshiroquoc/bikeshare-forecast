"""Replay consecutive days without allowing model look-ahead leakage."""

import argparse
from datetime import date, timedelta

import yaml

from src.serving.batch_predict import load_inputs, run_batch_predict
from src.serving.evaluate import evaluate_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model-dir", default="models/current")
    parser.add_argument("--out-dir", default="data/predictions")
    parser.add_argument(
        "--reference-mae",
        type=float,
        help="MAE from the matching backtest dates, not the all-window mean.",
    )
    args = parser.parse_args()

    if args.days < 1:
        raise ValueError("--days must be at least 1.")

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    history, weather, station_master = load_inputs()
    start = date.fromisoformat(args.start)

    for offset in range(args.days):
        run_batch_predict(
            as_of=start + timedelta(days=offset),
            history=history,
            weather=weather,
            station_master=station_master,
            cfg=config,
            model_dir=args.model_dir,
            out_dir=args.out_dir,
        )

    report = evaluate_predictions(args.out_dir, history)
    replay_mae = float(report["mae"].mean())
    print(report)
    print(f"\nMean replay MAE: {replay_mae:.3f}")
    if args.reference_mae is not None:
        delta = replay_mae - args.reference_mae
        print(
            f"Matching backtest MAE: {args.reference_mae:.3f} | "
            f"delta: {delta:+.3f}"
        )
    else:
        print(
            "Compare with the same dates from Week 3, not the four-window mean."
        )
