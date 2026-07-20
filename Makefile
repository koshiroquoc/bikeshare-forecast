.PHONY: ingest test lint features backtest mlflow-day1 mlflow-day4 mlflow-day5-tune mlflow-day5-final train-production forecast replay api monitor promote docker-build docker-up docker-down

export PREFECT_SERVER_EPHEMERAL_ENABLED=True

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

mlflow-day4:
	MLFLOW_TRACKING_URI=sqlite:///$(CURDIR)/mlflow.db PYTHONPATH=. uv run python scripts/run_day4_variants.py

mlflow-day5-tune:
	MLFLOW_TRACKING_URI=sqlite:///$(CURDIR)/mlflow.db PYTHONPATH=. uv run python scripts/run_day5_tuning.py

mlflow-day5-final:
	MLFLOW_TRACKING_URI=sqlite:///$(CURDIR)/mlflow.db PYTHONPATH=. uv run python scripts/run_day5_final.py

train-production:
	uv run python -m src.training.train_production $(if $(TRAIN_THROUGH),--train-through $(TRAIN_THROUGH),)

forecast:
	uv run python -m flows.nightly_forecast --as-of $(AS_OF) $(if $(MODEL_DIR),--model-dir $(MODEL_DIR),) $(if $(OUT_DIR),--out-dir $(OUT_DIR),)

replay:
	uv run python -m scripts.replay --start $(START) --days $(or $(DAYS),7) $(if $(MODEL_DIR),--model-dir $(MODEL_DIR),) $(if $(OUT_DIR),--out-dir $(OUT_DIR),) $(if $(REFERENCE_MAE),--reference-mae $(REFERENCE_MAE),)

api:
	uv run uvicorn src.serving.api:app --reload

monitor:
	uv run python -m scripts.run_monitoring --model-dir $(MODEL_DIR) $(if $(PREDICTIONS_DIR),--predictions-dir $(PREDICTIONS_DIR),) $(if $(REPORT_DIR),--report-dir $(REPORT_DIR),)

promote:
	uv run python -m src.training.promote $(if $(CURRENT_DIR),--current-dir $(CURRENT_DIR),) $(if $(ARCHIVE_DIR),--archive-dir $(ARCHIVE_DIR),) $(if $(MIN_IMPROVEMENT),--min-improvement $(MIN_IMPROVEMENT),)

docker-build:
	docker build --tag bikeshare-forecast:local .

docker-up:
	docker compose up --detach api

docker-down:
	docker compose down
