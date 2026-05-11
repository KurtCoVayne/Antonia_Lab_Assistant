.PHONY: dev-up jetson-up down run ingest bench lint typecheck test test-live install

dev-up:
	docker compose -f docker/docker-compose.yml up -d

jetson-up:
	docker compose -f docker/docker-compose.yml \
	               -f docker/docker-compose.jetson.yml up -d

down:
	docker compose -f docker/docker-compose.yml down

run:
	uv run antonia-run

ingest:
	uv run antonia-ingest

bench:
	uv run antonia-bench

lint:
	uv run ruff check src/ tests/ scripts/

typecheck:
	uv run mypy src/

test:
	uv run pytest tests/unit/ -v

test-live:
	ANTONIA_LIVE_TESTS=1 uv run pytest tests/integration/ -v

install:
	uv pip install -e ".[dev]"
