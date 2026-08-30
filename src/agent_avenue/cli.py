"""Developer CLI for seeded games, balanced arenas, and replay verification."""

import argparse
import json
from collections.abc import Iterator, Mapping
from pathlib import Path

from .agents import (
    SEED_DERIVATION,
    EpsilonConfig,
    EpsilonGreedyAgent,
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


def _agent_spec(kind: str, agent_id: str, checkpoint: Path | None = None) -> AgentSpec:
    if kind == "random":
        if checkpoint is not None:
            raise ValueError("checkpoint path is only valid for a learned agent")
        random_config = RandomAgentConfig()
        return AgentSpec(agent_id, random_config.to_data(), RandomAgent)
    if kind == "heuristic":
        if checkpoint is not None:
            raise ValueError("checkpoint path is only valid for a learned agent")
        heuristic_config = GreedyHeuristicConfig()
        return AgentSpec(agent_id, heuristic_config.to_data(), GreedyHeuristicAgent)
    if kind == "learned":
        if checkpoint is None:
            raise ValueError("learned agents require a checkpoint path")
        from .agents.learned import LearnedValueAgent
        from .learning import load_checkpoint

        loaded = load_checkpoint(checkpoint)
        agent = LearnedValueAgent.from_checkpoint(loaded)
        return AgentSpec(agent_id, agent.config.to_data(), lambda: agent)
    raise ValueError(f"unsupported agent kind: {kind}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    game = subparsers.add_parser("game", help="run one seeded agent-versus-agent game")
    game.add_argument("--seed", type=int, default=0, help="setup seed")
    game.add_argument(
        "--player-one", choices=("random", "heuristic", "learned"), default="heuristic"
    )
    game.add_argument("--player-two", choices=("random", "heuristic", "learned"), default="random")
    game.add_argument("--player-one-checkpoint", type=Path)
    game.add_argument("--player-two-checkpoint", type=Path)
    game.add_argument("--player-one-seed", type=int)
    game.add_argument("--player-two-seed", type=int)
    game.add_argument("--output", type=Path, help="write a completed-game JSON record")

    arena = subparsers.add_parser("arena", help="run a paired, seat-balanced deterministic arena")
    arena.add_argument("--agent-a", choices=("random", "heuristic", "learned"), default="heuristic")
    arena.add_argument("--agent-b", choices=("random", "heuristic", "learned"), default="random")
    arena.add_argument("--agent-a-checkpoint", type=Path)
    arena.add_argument("--agent-b-checkpoint", type=Path)
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

    corpus = subparsers.add_parser(
        "corpus-generate", help="generate a verified heuristic self-play corpus"
    )
    corpus.add_argument("output", type=Path)
    corpus.add_argument("--games", type=int, default=100)
    corpus.add_argument("--seed", type=int, default=0, help="corpus root seed")
    corpus.add_argument("--run-id", default="heuristic-bootstrap")

    dataset = subparsers.add_parser(
        "dataset-build", help="extract safe samples from a verified corpus"
    )
    dataset.add_argument("corpus", type=Path)
    dataset.add_argument("output", type=Path)
    dataset.add_argument("--split-seed", type=int, default=0)
    dataset.add_argument("--allow-code-mismatch", action="store_true")

    train = subparsers.add_parser("train", help="train candidate-mlp-v1 and save a checkpoint")
    train.add_argument("dataset", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--max-epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--cpu-threads", type=int, default=1)

    checkpoint = subparsers.add_parser(
        "checkpoint-inspect", help="validate and inspect an immutable neural checkpoint"
    )
    checkpoint.add_argument("path", type=Path)
    return parser


def _run_game_command(args: argparse.Namespace) -> dict[str, object]:
    first = _agent_spec(
        args.player_one,
        f"player-one-{args.player_one}",
        args.player_one_checkpoint,
    )
    second = _agent_spec(
        args.player_two,
        f"player-two-{args.player_two}",
        args.player_two_checkpoint,
    )
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
        _agent_spec(args.agent_a, f"agent-a-{args.agent_a}", args.agent_a_checkpoint),
        _agent_spec(args.agent_b, f"agent-b-{args.agent_b}", args.agent_b_checkpoint),
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


def _run_corpus_generate_command(args: argparse.Namespace) -> dict[str, object]:
    if args.games < 2:
        raise ValueError("corpus generation requires at least two games")
    from .runners import iter_games
    from .storage import write_corpus

    base_config = GreedyHeuristicConfig()
    epsilon = EpsilonConfig(1, 5)
    wrapped_config = {
        "version": "epsilon-wrapper-v1",
        "base": base_config.to_data(),
        "epsilon": epsilon.to_data(),
    }
    agent = AgentSpec(
        "epsilon-heuristic-v1",
        wrapped_config,
        lambda: EpsilonGreedyAgent(GreedyHeuristicAgent(), epsilon),
    )

    def specs() -> Iterator[GameSpec]:
        for index in range(args.games):
            domain = f"corpus:game:{index}"
            setup_seed = derive_seed(args.seed, f"{domain}:setup") & ((1 << 64) - 1)
            yield GameSpec(
                args.run_id,
                f"game-{index:06d}",
                None,
                GameConfig(),
                setup_seed,
                (agent, agent),
                (
                    derive_seed(args.seed, f"{domain}:player-one"),
                    derive_seed(args.seed, f"{domain}:player-two"),
                ),
                (SEED_DERIVATION, SEED_DERIVATION),
            )

    manifest = write_corpus(
        args.output,
        iter_games(specs()),
        run_id=args.run_id,
        behavior_policy="epsilon-1/5-greedy-public-v1",
        root_seed=args.seed,
        generation=0,
    )
    return manifest.to_data()


def _run_dataset_build_command(args: argparse.Namespace) -> dict[str, object]:
    from .learning.dataset import materialize_dataset, save_dataset
    from .storage import load_corpus

    manifest, records = load_corpus(args.corpus, verify_code=not args.allow_code_mismatch)
    dataset = materialize_dataset(
        records,
        split_seed=args.split_seed,
        verify_code=not args.allow_code_mismatch,
        source_corpus_fingerprint=manifest.corpus_fingerprint,
    )
    arrays, manifest_path = save_dataset(dataset, args.output)
    return {
        "dataset_fingerprint": dataset.fingerprint,
        "arrays": str(arrays),
        "manifest": str(manifest_path),
        "counts": dataset.manifest["counts"],
    }


def _run_train_command(args: argparse.Namespace) -> dict[str, object]:
    from .learning import TrainingConfig, load_dataset, save_checkpoint, train_model

    dataset = load_dataset(args.dataset)
    config = TrainingConfig(
        seed=args.seed,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        cpu_threads=args.cpu_threads,
    )
    result = train_model(dataset, config=config)
    source_corpus = dataset.manifest.get("source_corpus_fingerprint")
    source_corpora = (source_corpus,) if isinstance(source_corpus, str) else ()
    saved = save_checkpoint(
        args.output,
        result.model,
        metrics=result.metrics_dict(),
        training_config=config.normalized(),
        training_seeds={"root": args.seed},
        dataset_fingerprint=dataset.fingerprint,
        source_corpus_fingerprints=source_corpora,
    )
    return {
        "checkpoint_fingerprint": saved.checkpoint_fingerprint,
        "path": str(saved.path),
        "best_epoch": result.best_epoch,
        "validation_loss": result.validation_metrics.equal_game_loss,
    }


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def _run_checkpoint_inspect_command(args: argparse.Namespace) -> dict[str, object]:
    from .learning import inspect_checkpoint

    result = inspect_checkpoint(args.path)
    return {
        "checkpoint_fingerprint": result.checkpoint_fingerprint,
        "tensor_digest": result.tensor_digest,
        "path": str(result.path),
        "manifest": _json_value(result.manifest),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.command == "game":
        result = _run_game_command(args)
    elif args.command == "arena":
        result = _run_arena_command(args)
    elif args.command == "replay":
        result = _run_replay_command(args)
    elif args.command == "corpus-generate":
        result = _run_corpus_generate_command(args)
    elif args.command == "dataset-build":
        result = _run_dataset_build_command(args)
    elif args.command == "train":
        result = _run_train_command(args)
    else:
        result = _run_checkpoint_inspect_command(args)
    print(json.dumps(result, sort_keys=True))
