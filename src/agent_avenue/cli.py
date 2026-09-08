"""Developer CLI for seeded games, balanced arenas, and replay verification."""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from .agents import (
    SEED_DERIVATION,
    EpsilonConfig,
    GreedyHeuristicAgent,
    GreedyHeuristicConfig,
    RandomAgent,
    RandomAgentConfig,
    TerminalSafetyAgent,
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


def _agent_spec(
    kind: str,
    agent_id: str,
    checkpoint: Path | None = None,
    *,
    terminal_safety: bool = False,
) -> AgentSpec:
    if kind == "random":
        if checkpoint is not None:
            raise ValueError("checkpoint path is only valid for a learned agent")
        random_config = RandomAgentConfig()
        if terminal_safety:
            wrapped_random = TerminalSafetyAgent(RandomAgent(random_config))
            return AgentSpec(agent_id, wrapped_random.config_to_data(), lambda: wrapped_random)
        return AgentSpec(agent_id, random_config.to_data(), RandomAgent)
    if kind == "heuristic":
        if checkpoint is not None:
            raise ValueError("checkpoint path is only valid for a learned agent")
        heuristic_config = GreedyHeuristicConfig()
        if terminal_safety:
            wrapped_heuristic = TerminalSafetyAgent(GreedyHeuristicAgent(heuristic_config))
            return AgentSpec(
                agent_id, wrapped_heuristic.config_to_data(), lambda: wrapped_heuristic
            )
        return AgentSpec(agent_id, heuristic_config.to_data(), GreedyHeuristicAgent)
    if kind == "learned":
        if checkpoint is None:
            raise ValueError("learned agents require a checkpoint path")
        from .agents.learned import LearnedValueAgent
        from .learning import load_checkpoint

        loaded = load_checkpoint(checkpoint)
        learned = LearnedValueAgent.from_checkpoint(loaded)
        if terminal_safety:
            wrapped_learned = TerminalSafetyAgent(learned)
            return AgentSpec(agent_id, wrapped_learned.config_to_data(), lambda: wrapped_learned)
        return AgentSpec(agent_id, learned.config.to_data(), lambda: learned)
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
    game.add_argument("--player-one-terminal-safety", action="store_true")
    game.add_argument("--player-two-terminal-safety", action="store_true")
    game.add_argument("--player-one-seed", type=int)
    game.add_argument("--player-two-seed", type=int)
    game.add_argument("--output", type=Path, help="write a completed-game JSON record")

    arena = subparsers.add_parser("arena", help="run a paired, seat-balanced deterministic arena")
    arena.add_argument("--agent-a", choices=("random", "heuristic", "learned"), default="heuristic")
    arena.add_argument("--agent-b", choices=("random", "heuristic", "learned"), default="random")
    arena.add_argument("--agent-a-checkpoint", type=Path)
    arena.add_argument("--agent-b-checkpoint", type=Path)
    arena.add_argument("--agent-a-terminal-safety", action="store_true")
    arena.add_argument("--agent-b-terminal-safety", action="store_true")
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
        "corpus-generate", help="generate heuristic bootstrap or frozen-checkpoint self-play"
    )
    corpus.add_argument("output", type=Path)
    corpus.add_argument("--games", type=int, default=100)
    corpus.add_argument("--seed", type=int, default=0, help="corpus root seed")
    corpus.add_argument("--run-id", default="heuristic-bootstrap")
    corpus.add_argument(
        "--checkpoint", type=Path, help="frozen incumbent checkpoint for generation > 0"
    )
    corpus.add_argument("--generation", type=int, default=0)
    corpus.add_argument("--attempt-id", default="attempt-1")
    corpus.add_argument("--epsilon-numerator", type=int)
    corpus.add_argument("--epsilon-denominator", type=int)
    corpus.add_argument("--terminal-safety", action="store_true")

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
    train.add_argument("--parent-checkpoint", type=Path)
    train.add_argument("--generation", type=int)

    checkpoint = subparsers.add_parser(
        "checkpoint-inspect", help="validate and inspect an immutable neural checkpoint"
    )
    checkpoint.add_argument("path", type=Path)

    bootstrap = subparsers.add_parser(
        "bootstrap", help="run or resume generation-zero corpus, training, and validation"
    )
    bootstrap.add_argument("output", type=Path)
    bootstrap.add_argument("--attempt-id", required=True)
    bootstrap.add_argument("--experiment-id", required=True)
    bootstrap.add_argument("--seed", type=int, required=True)
    bootstrap.add_argument("--games", type=int, default=4000)
    bootstrap.add_argument("--max-epochs", type=int, default=50)
    bootstrap.add_argument("--batch-size", type=int, default=1024)
    bootstrap.add_argument("--learning-rate", type=float, default=1e-3)
    bootstrap.add_argument("--weight-decay", type=float, default=1e-4)
    bootstrap.add_argument("--patience", type=int, default=8)
    bootstrap.add_argument("--cpu-threads", type=int, default=1)
    bootstrap.add_argument("--terminal-safety", action="store_true")
    bootstrap.add_argument("--dry-run", action="store_true")

    safety_audit = subparsers.add_parser(
        "safety-audit", help="replay a retained corpus and audit immediate-loss decisions"
    )
    safety_audit.add_argument("corpus", type=Path)
    safety_audit.add_argument("--output", type=Path)
    safety_audit.add_argument("--allow-code-mismatch", action="store_true")

    diagnostics = subparsers.add_parser(
        "diagnostics",
        help="inventory and retrospectively audit retained M6/terminal-safety artifacts",
    )
    diagnostics.add_argument(
        "output", type=Path, help="directory for deterministic diagnostic files"
    )
    diagnostics.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="repository root containing ignored runs/, artifacts/, and checkpoints/",
    )
    diagnostics.add_argument(
        "--archive",
        type=Path,
        action="append",
        default=[],
        help="retained tar.gz archive to checksum and inspect; repeat for each archive",
    )
    diagnostics.add_argument(
        "--restore-directory",
        type=Path,
        help="empty disposable directory for archive extraction and semantic corpus checks",
    )
    diagnostics.add_argument(
        "--trace-limit", type=int, default=4, help="number of information-safe positions to emit"
    )

    crossplay = subparsers.add_parser(
        "crossplay-evaluate",
        help="evaluate qn on held-out records from every prior-policy pair",
    )
    crossplay.add_argument("output", type=Path)
    crossplay.add_argument("--candidate", type=Path, required=True)
    crossplay.add_argument("--candidate-label", required=True)
    crossplay.add_argument("--generation", type=int, required=True)
    crossplay.add_argument(
        "--prior",
        action="append",
        default=[],
        metavar="LABEL=CHECKPOINT",
        help="repeat in q0..q(n-1) order",
    )
    crossplay.add_argument("--exclude-corpus", action="append", type=Path, default=[])
    crossplay.add_argument("--pairs", type=int, default=200)
    crossplay.add_argument("--seed", type=int, required=True)
    crossplay.add_argument("--terminal-safety", action="store_true")
    crossplay.add_argument("--dry-run", action="store_true")

    iterate = subparsers.add_parser(
        "iterate", help="run or resume one complete frozen self-play generation"
    )
    iterate.add_argument("output", type=Path)
    iterate.add_argument("--incumbent", type=Path, required=True)
    iterate.add_argument("--generation", type=int, required=True)
    iterate.add_argument("--attempt-id", required=True)
    iterate.add_argument("--seed", type=int, required=True, help="iteration root seed")
    iterate.add_argument("--games", type=int, default=4000)
    iterate.add_argument("--primary-pairs", type=int, default=500)
    iterate.add_argument("--guardrail-pairs", type=int, default=200)
    iterate.add_argument("--confirmation-pairs", type=int, default=1000)
    iterate.add_argument("--max-epochs", type=int, default=50)
    iterate.add_argument("--batch-size", type=int, default=1024)
    iterate.add_argument("--learning-rate", type=float, default=1e-3)
    iterate.add_argument("--weight-decay", type=float, default=1e-4)
    iterate.add_argument("--patience", type=int, default=8)
    iterate.add_argument("--cpu-threads", type=int, default=1)
    iterate.add_argument("--dry-run", action="store_true")
    iterate.add_argument("--terminal-safety", action="store_true")
    return parser


def _run_game_command(args: argparse.Namespace) -> dict[str, object]:
    first = _agent_spec(
        args.player_one,
        f"player-one-{args.player_one}",
        args.player_one_checkpoint,
        terminal_safety=args.player_one_terminal_safety,
    )
    second = _agent_spec(
        args.player_two,
        f"player-two-{args.player_two}",
        args.player_two_checkpoint,
        terminal_safety=args.player_two_terminal_safety,
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
    from .runners import run_resumable_arena
    from .runners.safety_audit import audit_terminal_safety
    from .storage import load_corpus

    config = ArenaConfig(
        args.run_id,
        _agent_spec(
            args.agent_a,
            f"agent-a-{args.agent_a}",
            args.agent_a_checkpoint,
            terminal_safety=args.agent_a_terminal_safety,
        ),
        _agent_spec(
            args.agent_b,
            f"agent-b-{args.agent_b}",
            args.agent_b_checkpoint,
            terminal_safety=args.agent_b_terminal_safety,
        ),
        args.pairs,
        args.seed,
    )
    if args.records_dir is None:
        data = run_arena(config).to_data()
    else:
        run_directory = args.records_dir / args.run_id
        retained = run_resumable_arena(
            run_directory,
            config,
            generation=None,
            corpus_configuration={
                "version": "cli-retained-arena-v1",
                "run_id": args.run_id,
            },
        )
        manifest, records = load_corpus(run_directory)
        data = retained.report.to_data()
        data["records"] = {
            "path": str(run_directory),
            "corpus_fingerprint": manifest.corpus_fingerprint,
            "declaration_fingerprint": manifest.declaration_fingerprint,
            "record_count": manifest.record_count,
        }
        data["safety_diagnostics"] = audit_terminal_safety(
            records, source_corpus_fingerprint=manifest.corpus_fingerprint
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
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
    from .runners import (
        generation_config_from_agent,
        heuristic_bootstrap_agent,
        learned_self_play_agent,
        planned_epsilon,
        run_resumable_corpus,
        schedule_generation,
        schedule_heuristic_bootstrap,
    )

    if args.generation == 0:
        if args.checkpoint is not None:
            raise ValueError("generation zero uses the heuristic bootstrap, not a checkpoint")
        if (args.epsilon_numerator, args.epsilon_denominator) not in {(None, None), (1, 5)}:
            raise ValueError("generation zero epsilon is fixed at 1/5")
        epsilon = EpsilonConfig(1, 5)
        agent = heuristic_bootstrap_agent(terminal_safety=args.terminal_safety)
        scheduled_specs = tuple(
            schedule_heuristic_bootstrap(
                run_id=args.run_id,
                game_count=args.games,
                root_seed=args.seed,
                attempt_id=args.attempt_id,
                agent=agent,
            )
        )
        behavior_policy = (
            "terminal-safety-v1:epsilon-1/5-greedy-public-v1"
            if args.terminal_safety
            else "epsilon-1/5-greedy-public-v1"
        )
        configuration: dict[str, object] = {
            "version": "heuristic-bootstrap-corpus-v1",
            "attempt_id": args.attempt_id,
            "epsilon": epsilon.to_data(),
            "games": args.games,
            "policy_shield": "terminal-safety-v1" if args.terminal_safety else None,
        }
    else:
        if args.generation < 1 or args.checkpoint is None:
            raise ValueError("learned self-play requires generation > 0 and --checkpoint")
        if (args.epsilon_numerator is None) != (args.epsilon_denominator is None):
            raise ValueError("provide both epsilon numerator and denominator")
        epsilon = (
            planned_epsilon(args.generation)
            if args.epsilon_numerator is None
            else EpsilonConfig(args.epsilon_numerator, args.epsilon_denominator)
        )
        agent = learned_self_play_agent(
            args.checkpoint, epsilon, terminal_safety=args.terminal_safety
        )
        generation = generation_config_from_agent(
            generation=args.generation,
            run_id=args.run_id,
            game_count=args.games,
            root_seed=args.seed,
            agent=agent,
            epsilon=epsilon,
            attempt_id=args.attempt_id,
        )
        scheduled_specs = tuple(schedule_generation(generation, agent))
        behavior_policy = (
            f"{generation.fingerprint}:terminal-safety-v1:epsilon-frozen-incumbent-v1"
            if args.terminal_safety
            else f"{generation.fingerprint}:epsilon-frozen-incumbent-v1"
        )
        configuration = {"generation": generation.normalized()}

    manifest = run_resumable_corpus(
        args.output,
        scheduled_specs,
        behavior_policy=behavior_policy,
        root_seed=args.seed,
        generation=args.generation,
        configuration=configuration,
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
    from .learning import (
        TrainingConfig,
        load_checkpoint,
        load_dataset,
        save_checkpoint,
        train_model,
    )

    if (args.parent_checkpoint is None) != (args.generation is None):
        raise ValueError("--parent-checkpoint and --generation must be provided together")
    if args.generation is not None and args.generation < 1:
        raise ValueError("parent-initialized training requires generation at least one")
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
    parent = load_checkpoint(args.parent_checkpoint) if args.parent_checkpoint is not None else None
    initial_state = parent.model.state_dict() if parent is not None else None
    result = train_model(dataset, config=config, initial_state_dict=initial_state)
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
        parent_checkpoint=parent.checkpoint_fingerprint if parent is not None else None,
        generation=args.generation,
    )
    return {
        "checkpoint_fingerprint": saved.checkpoint_fingerprint,
        "path": str(saved.path),
        "best_epoch": result.best_epoch,
        "validation_loss": result.validation_metrics.equal_game_loss,
    }


def _run_bootstrap_command(args: argparse.Namespace) -> dict[str, object]:
    from .runners.bootstrap import (
        BootstrapConfig,
        resolve_bootstrap_plan,
        run_bootstrap,
    )

    plan = resolve_bootstrap_plan(
        BootstrapConfig(
            output=args.output,
            attempt_id=args.attempt_id,
            experiment_id=args.experiment_id,
            root_seed=args.seed,
            game_count=args.games,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            cpu_threads=args.cpu_threads,
            terminal_safety=args.terminal_safety,
        )
    )
    if args.dry_run:
        return plan.to_data()
    return run_bootstrap(plan).to_data()


def _run_safety_audit_command(args: argparse.Namespace) -> dict[str, object]:
    from .runners.safety_audit import audit_terminal_safety
    from .storage import load_corpus

    manifest, records = load_corpus(args.corpus, verify_code=not args.allow_code_mismatch)
    result = audit_terminal_safety(
        records,
        source_corpus_fingerprint=manifest.corpus_fingerprint,
        verify_code=not args.allow_code_mismatch,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def _run_diagnostics_command(args: argparse.Namespace) -> dict[str, object]:
    from .runners.diagnostics import run_diagnostics

    return run_diagnostics(
        artifact_root=args.artifact_root,
        output=args.output,
        archives=tuple(args.archive),
        restore_directory=args.restore_directory,
        trace_limit=args.trace_limit,
    )


def _parse_prior(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise ValueError("--prior must use LABEL=CHECKPOINT")
    return label, Path(raw_path)


def _run_crossplay_command(args: argparse.Namespace) -> dict[str, object]:
    from .runners.crossplay import CrossplayConfig, resolve_crossplay_plan, run_crossplay

    plan = resolve_crossplay_plan(
        CrossplayConfig(
            output=args.output,
            candidate_checkpoint=args.candidate,
            candidate_label=args.candidate_label,
            generation=args.generation,
            prior_checkpoints=tuple(_parse_prior(value) for value in args.prior),
            exclusion_corpora=tuple(args.exclude_corpus),
            root_seed=args.seed,
            pair_count=args.pairs,
            terminal_safety=args.terminal_safety,
        )
    )
    if args.dry_run:
        return plan.to_data()
    report_path = run_crossplay(plan)
    report = json.loads(report_path.read_text())
    if not isinstance(report, dict):  # pragma: no cover - runner writes an object
        raise ValueError("crossplay report must be an object")
    return report


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


def _run_iterate_command(args: argparse.Namespace) -> dict[str, object]:
    from .runners import IterationConfig, resolve_iteration_plan, run_iteration

    config = IterationConfig(
        output=args.output,
        incumbent_checkpoint=args.incumbent,
        generation=args.generation,
        attempt_id=args.attempt_id,
        root_seed=args.seed,
        game_count=args.games,
        primary_pairs=args.primary_pairs,
        guardrail_pairs=args.guardrail_pairs,
        confirmation_pairs=args.confirmation_pairs,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        cpu_threads=args.cpu_threads,
        terminal_safety=args.terminal_safety,
    )
    plan = resolve_iteration_plan(config)
    if args.dry_run:
        return plan.to_data()
    return run_iteration(plan).to_data()


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
    elif args.command == "checkpoint-inspect":
        result = _run_checkpoint_inspect_command(args)
    elif args.command == "bootstrap":
        result = _run_bootstrap_command(args)
    elif args.command == "safety-audit":
        result = _run_safety_audit_command(args)
    elif args.command == "diagnostics":
        result = _run_diagnostics_command(args)
    elif args.command == "crossplay-evaluate":
        result = _run_crossplay_command(args)
    else:
        result = _run_iterate_command(args)
    print(json.dumps(result, sort_keys=True))
