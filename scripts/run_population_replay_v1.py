#!/usr/bin/env python3
"""Run the frozen M7 population replay or an explicitly non-claim toy smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_avenue.runners.population_experiment import (
    DEFAULT_CHECKPOINT_PATHS,
    FIXED_CREATED_AT,
    HISTORICAL_Q0,
    POPULATION_POLICY_IDS,
    PopulationExperimentConfig,
    build_population_policy_bundle,
    run_population_experiment,
)


def _toy_checkpoint_paths(root: Path) -> dict[str, Path]:
    """Create deterministic tiny smoke parents without ever changing claim input defaults."""
    from agent_avenue.learning import create_model, save_checkpoint

    paths: dict[str, Path] = {}
    for index, policy_id in enumerate((*POPULATION_POLICY_IDS[:5], HISTORICAL_Q0), start=1):
        path = root / policy_id
        if not path.exists():
            model = create_model(70_000 + index)
            save_checkpoint(
                path,
                model,
                metrics={"smoke": True},
                training_config={"smoke": True},
                training_seeds={"seed": 70_000 + index},
                metadata={"kind": "population-replay-toy-input"},
                created_at=FIXED_CREATED_AT,
            )
        paths[policy_id] = path
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/m7-population-replay-v1"))
    parser.add_argument(
        "--toy-smoke",
        action="store_true",
        help="use generated toy parents and bounded non-claim corpus/arena schedules",
    )
    parser.add_argument("--smoke-pairs", type=int, default=2)
    parser.add_argument("--smoke-max-epochs", type=int, default=2)
    parser.add_argument(
        "--holdout-root",
        action="append",
        type=Path,
        help="holdout root; required to be isolated from real evidence for toy smoke",
    )
    args = parser.parse_args()
    if args.toy_smoke:
        paths = _toy_checkpoint_paths(args.output / "smoke-inputs")
        roots = tuple(args.holdout_root) if args.holdout_root else (args.output / "smoke-holdout",)
        config = PopulationExperimentConfig(
            output=args.output,
            checkpoint_paths=paths,
            holdout_roots=roots,
            smoke_pair_count=args.smoke_pairs,
            smoke_max_epochs=args.smoke_max_epochs,
        )
        bundle = build_population_policy_bundle(paths, toy=True)
    else:
        if args.holdout_root:
            parser.error("--holdout-root is permitted only with --toy-smoke")
        if args.smoke_pairs != 2 or args.smoke_max_epochs != 2:
            parser.error("smoke options require --toy-smoke")
        config = PopulationExperimentConfig(
            output=args.output, checkpoint_paths=DEFAULT_CHECKPOINT_PATHS
        )
        bundle = None
    result = run_population_experiment(config, bundle)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - command boundary
        print(f"population replay failed: {exc}", file=sys.stderr)
        raise
