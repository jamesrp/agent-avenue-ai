.PHONY: setup run web test test-web test-rl lint format typecheck check check-rl arena-smoke neural-smoke clean

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

test-rl:
	uv run --extra rl pytest tests/encoding tests/learning

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy

check: lint typecheck test

check-rl:
	uv run --extra rl ruff check .
	uv run --extra rl mypy
	uv run --extra rl pytest

arena-smoke:
	uv run python -m agent_avenue arena --agent-a random --agent-b random --pairs 10 --seed 20260829 --run-id smoke-random-fairness
	uv run python -m agent_avenue arena --agent-a heuristic --agent-b random --pairs 10 --seed 20260829 --run-id smoke-heuristic-baseline

neural-smoke:
	rm -rf /tmp/agent-avenue-neural-smoke
	uv run --extra rl python -m agent_avenue corpus-generate /tmp/agent-avenue-neural-smoke/corpus --games 20 --seed 20260829 --run-id neural-smoke
	uv run --extra rl python -m agent_avenue dataset-build /tmp/agent-avenue-neural-smoke/corpus /tmp/agent-avenue-neural-smoke/dataset.npz --split-seed 17
	uv run --extra rl python -m agent_avenue train /tmp/agent-avenue-neural-smoke/dataset.npz /tmp/agent-avenue-neural-smoke/checkpoint --seed 23 --max-epochs 3 --batch-size 64
	uv run --extra rl python -m agent_avenue checkpoint-inspect /tmp/agent-avenue-neural-smoke/checkpoint

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
