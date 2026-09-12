#!/usr/bin/env python3
"""Dispatch an explicit frozen Step-4 claim or one bounded nonclaim smoke."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from agent_avenue.runners.rollout_experiment import (
    CYCLE_ID,
    RolloutExperimentConfig,
    run_rollout_experiment,
)
from agent_avenue.storage import repository_root


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--claim", action="store_true", help="run only frozen claim defaults")
    mode.add_argument("--smoke", action="store_true", help="run the bounded nonclaim smoke")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--step3-root", type=Path)
    parser.add_argument("--validator-preflight", type=Path)
    parser.add_argument(
        "--smoke-treatment-epochs",
        type=int,
        default=1,
        help="nonclaim treatment horizon; controls still reproduce retained full horizons",
    )
    args = parser.parse_args()
    root = repository_root()
    exact_step3 = root / "runs" / "m7-structured-model-v2"
    exact_output = root / "runs" / CYCLE_ID
    if args.claim:
        if (
            args.output is not None
            or args.step3_root is not None
            or args.holdout_root
            or args.smoke_treatment_epochs != 1
            or args.validator_preflight is None
        ):
            parser.error(
                "--claim fixes output, Step-3 root, holdout roots, and has no smoke options"
            )
        config = RolloutExperimentConfig(
            output=exact_output,
            step3_root=exact_step3,
            holdout_roots=(root / "runs",),
            validator_preflight=args.validator_preflight,
            claim=True,
        )
    else:
        if args.validator_preflight is not None:
            parser.error("--validator-preflight is only valid with --claim")
        if args.step3_root is None:
            parser.error("--smoke requires --step3-root retained input")
        output = args.output or Path(tempfile.mkdtemp(prefix="agent-avenue-step4-smoke-"))
        if output.resolve().is_relative_to((root / "runs").resolve()):
            parser.error("smoke output must be isolated outside repository runs")
        holdouts = tuple(args.holdout_root) if args.holdout_root else (output / "holdout",)
        config = RolloutExperimentConfig(
            output=output,
            step3_root=args.step3_root,
            holdout_roots=holdouts,
            claim=False,
            smoke_positions_per_stratum=1,
            smoke_treatment_epochs=args.smoke_treatment_epochs,
            smoke_arena_pairs=1,
        )
    print(json.dumps(run_rollout_experiment(config), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
