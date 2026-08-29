.PHONY: setup run test lint format typecheck check clean

setup:
	uv sync

run:
	uv run python -m agent_avenue

test:
	uv run pytest

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
