#!/usr/bin/env python3
"""Dispatch either an explicit frozen Step-3 claim or an isolated bounded smoke."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from agent_avenue.runners.structured_experiment import (
    CYCLE_ID,
    StructuredExperimentConfig,
    run_structured_experiment,
)
from agent_avenue.storage import repository_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", action="store_true", help="run only the frozen claim defaults")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--step2-root", type=Path)
    parser.add_argument("--smoke-pairs", type=int)
    parser.add_argument("--smoke-max-epochs", type=int)
    parser.add_argument("--holdout-root", action="append", type=Path)
    args = parser.parse_args()
    root = repository_root()
    exact_step2 = root / "runs/m7-population-replay-v1"
    exact_output = root / "runs" / CYCLE_ID
    if args.claim:
        if args.output is not None or args.step2_root is not None or args.holdout_root:
            parser.error("--claim fixes output, Step-2 root, and repository runs holdout")
        if args.smoke_pairs is not None or args.smoke_max_epochs is not None:
            parser.error("--claim cannot be combined with smoke truncation options")
        config = StructuredExperimentConfig(
            output=exact_output,
            step2_root=exact_step2,
            holdout_roots=(root / "runs",),
        )
    else:
        if args.step2_root is None:
            parser.error("non-claim smoke requires --step2-root")
        output = args.output or Path(tempfile.mkdtemp(prefix="agent-avenue-structured-v2-smoke-"))
        if output.resolve().is_relative_to((root / "runs").resolve()):
            parser.error("smoke output must be isolated outside repository runs")
        holdouts = tuple(args.holdout_root) if args.holdout_root else (output / "holdout",)
        config = StructuredExperimentConfig(
            output=output,
            step2_root=args.step2_root,
            holdout_roots=holdouts,
            smoke_pairs=args.smoke_pairs if args.smoke_pairs is not None else 1,
            smoke_max_epochs=(args.smoke_max_epochs if args.smoke_max_epochs is not None else 1),
        )
    print(json.dumps(run_structured_experiment(config), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
