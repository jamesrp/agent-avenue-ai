#!/usr/bin/env python3
"""Run the approved Step-5 independent league, its nonclaim smoke, or the runtime preflight.

Claim execution requires the committed input registry, retained learned checkpoints, a
tracked-clean source tree, output ``runs/m7-independent-league-promotion-v1``, holdout scope
``runs``, and a passing runtime preflight measured from a validated learned-checkpoint smoke.
The runner never changes the selected champion or the web default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_avenue.runners.league import (
    CYCLE_ID,
    REGISTRY_PATH,
    LeagueRunConfig,
    build_runtime_preflight,
    run_league,
    write_immutable,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the claim league or a bounded nonclaim smoke")
    run.add_argument("--output", type=Path, default=Path("runs") / CYCLE_ID)
    run.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    run.add_argument("--inputs-root", type=Path, default=Path("."))
    run.add_argument(
        "--policy-mode",
        choices=("learned-checkpoints", "toy-random"),
        default="learned-checkpoints",
    )
    run.add_argument(
        "--smoke-pairs",
        type=int,
        default=None,
        help="nonclaim smoke on separate seed domains with this many blocks per family",
    )
    run.add_argument("--holdout-root", type=Path, action="append", default=None)
    run.add_argument("--preflight", type=Path, default=None)
    preflight = commands.add_parser(
        "preflight", help="project claim runtime from a validated learned smoke"
    )
    preflight.add_argument("smoke_output", type=Path)
    preflight.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "preflight":
        artifact = build_runtime_preflight(arguments.smoke_output)
        write_immutable(arguments.output, artifact)
        print(json.dumps(artifact, sort_keys=True, indent=2))
        return 0 if artifact["claim_eligible"] is True else 3
    if arguments.smoke_pairs is not None and arguments.smoke_pairs < 1:
        raise SystemExit("--smoke-pairs must be positive")
    result = run_league(
        LeagueRunConfig(
            output=arguments.output,
            registry_path=arguments.registry,
            inputs_root=arguments.inputs_root,
            policy_mode=arguments.policy_mode,
            smoke_pairs=arguments.smoke_pairs,
            holdout_roots=tuple(arguments.holdout_root or (Path("runs"),)),
            preflight_path=arguments.preflight,
        )
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
