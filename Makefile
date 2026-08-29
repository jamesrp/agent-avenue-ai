.PHONY: setup run web test test-web lint format typecheck check arena-smoke clean

setup:
	uv sync

run:
	uv run python -m agent_avenue game

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

arena-smoke:
	uv run python -m agent_avenue arena --agent-a random --agent-b random --pairs 10 --seed 20260829 --run-id smoke-random-fairness
	uv run python -m agent_avenue arena --agent-a heuristic --agent-b random --pairs 10 --seed 20260829 --run-id smoke-heuristic-baseline

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
