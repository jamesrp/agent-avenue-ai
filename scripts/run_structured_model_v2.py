#!/usr/bin/env python3
"""Run only an explicitly bounded non-claim Step-3 structured-model smoke."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from agent_avenue.runners.structured_experiment import (
    StructuredExperimentConfig,
    run_structured_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--step2-root",
        type=Path,
        required=True,
        help="retained runs/m7-population-replay-v1 root",
    )
    parser.add_argument("--smoke-pairs", type=int, default=1)
    parser.add_argument("--smoke-max-epochs", type=int, default=1)
    parser.add_argument("--holdout-root", action="append", type=Path)
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="agent-avenue-structured-v2-smoke-"))
    if output.resolve().is_relative_to(Path("runs").resolve()):
        parser.error("smoke output must be isolated outside repository runs")
    holdouts = tuple(args.holdout_root) if args.holdout_root else (output / "holdout",)
    config = StructuredExperimentConfig(
        output=output,
        step2_root=args.step2_root,
        holdout_roots=holdouts,
        smoke_pairs=args.smoke_pairs,
        smoke_max_epochs=args.smoke_max_epochs,
    )
    print(json.dumps(run_structured_experiment(config), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
