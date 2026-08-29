"""Developer CLI for seeded games, balanced arenas, and replay verification."""

import argparse
import json
from pathlib import Path

from .agents import (
    SEED_DERIVATION,
    GreedyHeuristicAgent,
    GreedyHeuristicConfig,
    RandomAgent,
    RandomAgentConfig,
    derive_seed,
)
from .engine import GameConfig, load_replay, replay, state_fingerprint
from .runners import AgentSpec, ArenaConfig, GameSpec, run_arena, run_game
from .storage import (
    GameRecordError,
    game_record_fingerprint,
    game_record_to_data,
    load_game_record,
    save_game_record,
    verify_game_record,
)


def _agent_spec(kind: str, agent_id: str) -> AgentSpec:
    if kind == "random":
        random_config = RandomAgentConfig()
        return AgentSpec(agent_id, random_config.to_data(), RandomAgent)
    if kind == "heuristic":
        heuristic_config = GreedyHeuristicConfig()
        return AgentSpec(agent_id, heuristic_config.to_data(), GreedyHeuristicAgent)
    raise ValueError(f"unsupported agent kind: {kind}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    game = subparsers.add_parser("game", help="run one seeded agent-versus-agent game")
    game.add_argument("--seed", type=int, default=0, help="setup seed")
    game.add_argument("--player-one", choices=("random", "heuristic"), default="heuristic")
    game.add_argument("--player-two", choices=("random", "heuristic"), default="random")
    game.add_argument("--player-one-seed", type=int)
    game.add_argument("--player-two-seed", type=int)
    game.add_argument("--output", type=Path, help="write a completed-game JSON record")

    arena = subparsers.add_parser("arena", help="run a paired, seat-balanced deterministic arena")
    arena.add_argument("--agent-a", choices=("random", "heuristic"), default="heuristic")
    arena.add_argument("--agent-b", choices=("random", "heuristic"), default="random")
    arena.add_argument("--pairs", type=int, default=10)
    arena.add_argument("--seed", type=int, default=0, help="arena master seed")
    arena.add_argument("--run-id", default="cli-arena")
    arena.add_argument("--output", type=Path, help="write the aggregate report JSON")
    arena.add_argument("--records-dir", type=Path, help="write individual game records")

    verify = subparsers.add_parser("replay", help="verify a game-record or engine-replay JSON file")
    verify.add_argument("path", type=Path)
    verify.add_argument(
        "--allow-code-mismatch",
        action="store_true",
        help="verify replay semantics while ignoring the source fingerprint",
    )
    return parser


def _run_game_command(args: argparse.Namespace) -> dict[str, object]:
    first = _agent_spec(args.player_one, f"player-one-{args.player_one}")
    second = _agent_spec(args.player_two, f"player-two-{args.player_two}")
    first_seed = args.player_one_seed
    first_derivation = "supplied"
    if first_seed is None:
        first_seed = derive_seed(args.seed, "cli-game:player-one")
        first_derivation = SEED_DERIVATION
    second_seed = args.player_two_seed
    second_derivation = "supplied"
    if second_seed is None:
        second_seed = derive_seed(args.seed, "cli-game:player-two")
        second_derivation = SEED_DERIVATION
    spec = GameSpec(
        "cli-game",
        f"seed-{args.seed}",
        None,
        GameConfig(),
        args.seed,
        (first, second),
        (first_seed, second_seed),
        (first_derivation, second_derivation),
    )
    record = run_game(spec)
    if args.output is not None:
        save_game_record(record, args.output)
    return game_record_to_data(record)


def _run_arena_command(args: argparse.Namespace) -> dict[str, object]:
    config = ArenaConfig(
        args.run_id,
        _agent_spec(args.agent_a, f"agent-a-{args.agent_a}"),
        _agent_spec(args.agent_b, f"agent-b-{args.agent_b}"),
        args.pairs,
        args.seed,
    )
    sink = None
    if args.records_dir is not None:
        run_directory = args.records_dir / args.run_id
        run_directory.mkdir(parents=True, exist_ok=True)

        def save_record(record):  # type: ignore[no-untyped-def]
            save_game_record(record, run_directory / f"{record.game_id}.game.json")

        sink = save_record
    data = run_arena(config, sink).to_data()
    if args.output is not None:
        args.output.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    return data


def _run_replay_command(args: argparse.Namespace) -> dict[str, object]:
    try:
        record = load_game_record(args.path)
    except GameRecordError:
        state = replay(load_replay(args.path))
        return {
            "verified": True,
            "format": "engine_replay",
            "fingerprint": state_fingerprint(state),
            "winner": state.outcome.winner.value if state.outcome else None,
        }
    state = verify_game_record(record, verify_code=not args.allow_code_mismatch)
    return {
        "verified": True,
        "format": "game_record",
        "record_fingerprint": game_record_fingerprint(record),
        "state_fingerprint": state_fingerprint(state),
        "winner": record.winner.value,
        "terminal_reason": record.terminal_reason,
    }


def main() -> None:
    args = _parser().parse_args()
    if args.command == "game":
        result = _run_game_command(args)
    elif args.command == "arena":
        result = _run_arena_command(args)
    else:
        result = _run_replay_command(args)
    print(json.dumps(result, sort_keys=True))
