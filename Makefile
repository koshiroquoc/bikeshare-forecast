.PHONY: ingest test lint features backtest

ingest:
	uv run python -m src.ingestion.divvy --config config/config.yaml
	uv run python -m src.ingestion.weather --mode historical --config config/config.yaml

test:
	uv run pytest -q

lint:
	uv run ruff check .

features:
	uv run python -m src.processing.build_features --config config/config.yaml

backtest:
	uv run python -c "import polars as pl, yaml; from src.training import backtest; cfg = yaml.safe_load(open('config/config.yaml')); cols = ['station_id', 'hour', 'trips', 'lag_168', 'day_of_week', 'hour_of_day']; df = pl.read_parquet('data/processed/features.parquet', columns=cols); results = backtest.run_backtest(df, [backtest.SeasonalNaive(), backtest.HistoricalMean()], n_windows=cfg['backtest']['n_windows']); print(results); print(backtest.summarize(results))"

mlflow-day1:
	PYTHONPATH=. uv run python scripts/run_day1.py