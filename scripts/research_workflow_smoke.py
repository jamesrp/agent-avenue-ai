#!/usr/bin/env python3
"""Deterministic analysis, review, and briefing steps for the workflow smoke cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def analyze(arena_path: Path, output: Path) -> None:
    arena = _read(arena_path)
    games = int(arena["game_count"])
    wins = int(arena["agent_a_wins"])
    pairs = arena["paired_seed_outcomes"]
    if not isinstance(pairs, list) or games != 2 * len(pairs):
        raise ValueError("arena does not contain complete paired, seat-swapped blocks")
    interval = arena["paired_bootstrap_confidence_interval_95"]
    _write(
        output,
        {
            "status": "completed",
            "evidence_class": "smoke_only",
            "games": games,
            "paired_blocks": len(pairs),
            "agent_a_wins": wins,
            "agent_a_win_rate": wins / games,
            "paired_bootstrap_interval_95": interval["interval"],
            "interpretation": (
                "Pipeline evidence only; four paired blocks support no strength claim."
            ),
        },
    )


def review(arena_path: Path, analysis_path: Path, output: Path) -> None:
    arena = _read(arena_path)
    analysis = _read(analysis_path)
    problems: list[str] = []
    if analysis.get("evidence_class") != "smoke_only":
        problems.append("analysis did not label the result smoke-only")
    if analysis.get("games") != arena.get("game_count"):
        problems.append("analysis game count differs from arena report")
    expected_rate = int(arena["agent_a_wins"]) / int(arena["game_count"])
    if analysis.get("agent_a_win_rate") != expected_rate:
        problems.append("analysis win rate is not reproducible from the arena report")
    pairs = arena.get("paired_seed_outcomes")
    if not isinstance(pairs, list) or len(pairs) != 4:
        problems.append("expected exactly four disposable paired blocks")
    _write(
        output,
        {
            "status": "completed" if not problems else "failed",
            "accepted": not problems,
            "checks": {
                "smoke_label": analysis.get("evidence_class") == "smoke_only",
                "game_count_recomputed": analysis.get("games") == arena.get("game_count"),
                "win_rate_recomputed": analysis.get("agent_a_win_rate") == expected_rate,
                "paired_block_count": len(pairs) if isinstance(pairs, list) else None,
            },
            "problems": problems,
        },
    )
    if problems:
        raise ValueError("; ".join(problems))


def brief(analysis_path: Path, review_path: Path, output: Path) -> None:
    analysis = _read(analysis_path)
    review_data = _read(review_path)
    if review_data.get("accepted") is not True:
        raise ValueError("review did not accept the smoke analysis")
    interval = analysis["paired_bootstrap_interval_95"]
    text = f"""# Research workflow setup smoke briefing

## 1. What we learned

The approved task graph automatically advanced from a real paired arena through deterministic
analysis, independent recomputation, and this briefing. This tests wiring, not playing strength.

## 2. Results and uncertainty

- Disposable games: {analysis["games"]} in {analysis["paired_blocks"]} seat-swapped blocks.
- Agent-A win rate: {analysis["agent_a_win_rate"]:.3f}.
- Paired-bootstrap interval: {interval[0]:.3f}-{interval[1]:.3f}.
- Evidence class: **smoke only**; the sample is intentionally too small for a scientific claim.

## 3. Failures, repairs, and deviations

No stage in this successful-path smoke run required a retry. Controlled retry, interruption,
stop, and wall-time behavior are exercised separately by the workflow verification tests.

## 4. Scientific limitations

Random-versus-random over four paired seeds is only a symmetry/wiring check. It says nothing about
learned-agent strength, training stability, or the current champion.

## 5. Board-game ML explanation

Seat-swapping reuses one setup seed with the logical agents reversed. This controls some luck from
the shuffled deck: the pair, rather than either single game, is the basic comparison unit.

## 6. Next decision

Do not launch another cycle automatically. Await explicit approval of the first real research-cycle
agreement.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("arena", type=Path)
    analyze_parser.add_argument("output", type=Path)
    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("arena", type=Path)
    review_parser.add_argument("analysis", type=Path)
    review_parser.add_argument("output", type=Path)
    brief_parser = subparsers.add_parser("brief")
    brief_parser.add_argument("analysis", type=Path)
    brief_parser.add_argument("review", type=Path)
    brief_parser.add_argument("output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "analyze":
        analyze(args.arena, args.output)
    elif args.command == "review":
        review(args.arena, args.analysis, args.output)
    else:
        brief(args.analysis, args.review, args.output)


if __name__ == "__main__":
    main()
