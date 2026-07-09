.PHONY: ingest test lint

ingest:
	uv run python -m src.ingestion.divvy --config config/config.yaml
	uv run python -m src.ingestion.weather --mode historical --config config/config.yaml

test:
	uv run pytest -q

lint:
	uv run ruff check .

products:
	uv run python -c "from pathlib import Path; import polars as pl, yaml; from src.processing import aggregate, cleaning; cfg = yaml.safe_load(open('config/config.yaml')); out_dir = Path('data/processed'); out_dir.mkdir(parents=True, exist_ok=True); trips = pl.scan_parquet('data/raw/divvy/*.parquet'); mapping = pl.read_csv('data/reference/station_mapping.csv'); clean = cleaning.clean_trips(trips, cfg, mapping); aggregate.build_station_master(clean).write_parquet(out_dir / 'station_master.parquet'); aggregate.build_station_month_panel(clean).write_parquet(out_dir / 'station_month_panel.parquet')"