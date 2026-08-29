.PHONY: setup run web test test-web lint format typecheck check clean

setup:
	uv sync

run:
	uv run python -m agent_avenue

web:
	uv run python -m agent_avenue.web

test:
	uv run pytest

test-web:
	uv run pytest tests/web

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy

check: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
