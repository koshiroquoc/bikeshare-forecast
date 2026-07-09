.PHONY: ingest test lint features

ingest:
	uv run python -m src.ingestion.divvy --config config/config.yaml
	uv run python -m src.ingestion.weather --mode historical --config config/config.yaml

test:
	uv run pytest -q

lint:
	uv run ruff check .

features:
	uv run python -m src.processing.build_features --config config/config.yaml