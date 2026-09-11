#!/usr/bin/env python3
"""Independently validate frozen M7 population replay evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_avenue.runners.population_experiment import validate_population_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/m7-population-replay-v1"))
    parser.add_argument(
        "--allow-smoke", action="store_true", help="allow a clearly labelled bounded smoke plan"
    )
    args = parser.parse_args()
    result = validate_population_experiment(args.output, allow_smoke=args.allow_smoke)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - command boundary
        print(f"population replay validation failed: {exc}", file=sys.stderr)
        raise
