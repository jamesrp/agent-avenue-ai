"""Developer CLI for deterministic demo games and replay verification."""

import argparse
import json
from pathlib import Path

from .engine import (
    GameState,
    Phase,
    create_replay,
    legal_actions,
    load_replay,
    new_game,
    replay,
    save_replay,
    state_fingerprint,
)
from .engine.replay import replay_to_data
from .engine.transitions import apply_action


def _run_scripted(seed: int) -> GameState:
    state = new_game(seed=seed)
    while state.phase is not Phase.TERMINAL:
        actions = legal_actions(state)
        if not actions:
            raise RuntimeError("nonterminal game has no legal action")
        state = apply_action(state, actions[0])
    return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    demo = subparsers.add_parser("demo", help="run a deterministic first-legal-action game")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--save", type=Path)
    verify = subparsers.add_parser("replay", help="verify a replay JSON file")
    verify.add_argument("path", type=Path)
    return parser


def main() -> None:
    """Run the requested developer verification command."""
    args = _parser().parse_args()
    if args.command == "replay":
        state = replay(load_replay(args.path))
        print(
            json.dumps(
                {
                    "verified": True,
                    "fingerprint": state_fingerprint(state),
                    "winner": state.outcome.winner.value if state.outcome else None,
                },
                sort_keys=True,
            )
        )
        return
    seed = args.seed if args.command == "demo" else 0
    state = _run_scripted(seed)
    record = create_replay(state.config, state.seed, state)
    if args.command == "demo" and args.save is not None:
        save_replay(record, args.save)
    print(json.dumps(replay_to_data(record), sort_keys=True))


if __name__ == "__main__":
    main()
