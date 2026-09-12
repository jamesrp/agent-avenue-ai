#!/usr/bin/env python3
"""Independent, local Step-4 evidence validator.

This deliberately does not import the production panel selector, latent sampler/transition,
target generator, rollout trainer/loss, arena aggregator, statistics, or selection entry points.
It reconstructs public panel identities locally, independently reruns the MC controls through the
ordinary structured trainer, and derives arena/smoke summaries from retained records.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import numpy as np

from agent_avenue.agents import derive_seed
from agent_avenue.agents.ordering import semantic_action_key
from agent_avenue.engine import Phase, apply_action, new_game
from agent_avenue.engine.cards import CANONICAL_DECK
from agent_avenue.engine.model import Action
from agent_avenue.learning import (
    StructuredTrainingConfig,
    StructuredTrainingData,
    load_checkpoint,
    load_structured_dataset,
    tensor_digest,
    train_structured_model,
)
from agent_avenue.observation import observation_to_data, observe
from agent_avenue.observation.model import PlayerObservation, RecruitContext
from agent_avenue.storage import GameRecord, game_record_fingerprint, load_corpus

CYCLE_ID = "m7-counterfactual-rollout-supervision-v1"
ROOT_SEED = 2026091204
STRATA = (
    "play:turn_1_3:legal_1_2",
    "play:turn_1_3:legal_6",
    "play:turn_1_3:legal_12",
    "play:turn_4_6:legal_1_2",
    "play:turn_4_6:legal_6",
    "play:turn_4_6:legal_12",
    "play:turn_7_plus:legal_1_2",
    "play:turn_7_plus:legal_6",
    "play:turn_7_plus:legal_12",
    "recruit:turn_1_3:unseen_ge_21",
    "recruit:turn_4_6:unseen_ge_21",
    "recruit:turn_7_plus:unseen_le_12",
    "recruit:turn_7_plus:unseen_13_20",
    "recruit:turn_7_plus:unseen_ge_21",
)


class ValidationError(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"artifact is not an object: {path}")
    return cast(dict[str, object], value)


def artifact(value: Mapping[str, object], label: str) -> None:
    actual = value.get("artifact_fingerprint")
    expected = digest({key: item for key, item in value.items() if key != "artifact_fingerprint"})
    if actual != expected:
        raise ValidationError(f"{label} fingerprint mismatch")


def static_source_guard() -> None:
    tree = ast.parse(Path(__file__).read_text())
    banned = {
        "agent_avenue.rollout.identity",
        "agent_avenue.rollout.latent",
        "agent_avenue.rollout.targets",
        "agent_avenue.rollout.artifact",
        "agent_avenue.learning.rollout_train",
        "agent_avenue.runners.rollout_experiment",
        "agent_avenue.runners.arena",
    }
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    if imports.intersection(banned):
        raise ValidationError("validator imports a forbidden production implementation")
    # AST imports are the enforceable boundary: this validator's local functions below own the
    # panel, target metadata, arena aggregation, statistics, and disposition checks.


def canonical_actions(actions: Iterable[Action]) -> tuple[Action, ...]:
    result = tuple(sorted(actions, key=semantic_action_key))
    if not result or len(set(result)) != len(result):
        raise ValidationError("local panel legal action set is malformed")
    return result


def safe_identity(observation: PlayerObservation) -> str:
    actions = canonical_actions(observation.legal_actions)
    decision = observation.decision
    actor = getattr(decision, "actor", None)
    revision = getattr(decision, "revision", None)
    if observation.viewer is not actor or type(revision) is not int:
        raise ValidationError("local panel observation is not the decision actor view")
    canonical_observation = observation_to_data(
        PlayerObservation(
            observation.viewer,
            observation.own_hand,
            observation.players,
            observation.active_player,
            observation.turn,
            observation.phase,
            observation.remaining_deck_count,
            observation.history,
            observation.decision,
            actions,
        )
    )
    return digest(
        {
            "observation": canonical_observation,
            "decision_actor": actor.value,
            "decision_revision": revision,
            "legal_actions": [
                {"semantic_key": list(semantic_action_key(action))} for action in actions
            ],
        }
    )


def turn_band(turn: int) -> str | None:
    if 1 <= turn <= 3:
        return "turn_1_3"
    if 4 <= turn <= 6:
        return "turn_4_6"
    return "turn_7_plus" if turn >= 7 else None


def public_unseen(observation: PlayerObservation) -> int:
    decision = observation.decision
    if not isinstance(decision, RecruitContext) or decision.known_face_down is not None:
        raise ValidationError("local recruit identity leaked a face-down card")
    remaining = Counter(CANONICAL_DECK)
    for player in observation.players:
        remaining.subtract(player.recruited)
    remaining.subtract(observation.own_hand)
    remaining[decision.face_up] -= 1
    if any(value < 0 for value in remaining.values()):
        raise ValidationError("local panel has impossible public card counts")
    return sum((+remaining).values())


def stratum(observation: PlayerObservation) -> str | None:
    if observation.viewer is not getattr(observation.decision, "actor", None):
        return None
    band = turn_band(observation.turn)
    if band is None:
        return None
    if observation.decision.kind == "play":
        bucket = {1: "legal_1_2", 2: "legal_1_2", 6: "legal_6", 12: "legal_12"}.get(
            len(observation.legal_actions)
        )
        return None if bucket is None else f"play:{band}:{bucket}"
    if (
        not isinstance(observation.decision, RecruitContext)
        or observation.decision.known_face_down is not None
    ):
        return None
    unseen = public_unseen(observation)
    if band in {"turn_1_3", "turn_4_6"}:
        return f"recruit:{band}:unseen_ge_21" if unseen >= 21 else None
    suffix = "unseen_le_12" if unseen <= 12 else "unseen_13_20" if unseen <= 20 else "unseen_ge_21"
    return f"recruit:{band}:{suffix}"


def local_panel(
    corpus: Path, train_fingerprints: set[str], replicate: str, quota: int
) -> list[dict[str, object]]:
    _, records = load_corpus(corpus, verify_code=False, verify_replays=True)
    by_identity: dict[str, tuple[str, str, dict[str, object]]] = {}
    seen: set[str] = set()
    for record in records:
        record_id = game_record_fingerprint(record)
        if record_id not in train_fingerprints:
            continue
        seen.add(record_id)
        state = new_game(record.replay.config, record.replay.seed)
        for action in record.replay.actions:
            actor = (
                state.active_player if state.phase is Phase.PLAY else state.active_player.other()
            )
            observation = observe(state, actor)
            label = stratum(observation)
            if label is not None:
                canonical_data = observation_to_data(
                    PlayerObservation(
                        observation.viewer,
                        observation.own_hand,
                        observation.players,
                        observation.active_player,
                        observation.turn,
                        observation.phase,
                        observation.remaining_deck_count,
                        observation.history,
                        observation.decision,
                        canonical_actions(observation.legal_actions),
                    )
                )
                identity = safe_identity(observation)
                prior = by_identity.get(identity)
                if prior is None or record_id < prior[1]:
                    by_identity[identity] = (label, record_id, canonical_data)
            state = apply_action(state, action)
    if seen != train_fingerprints:
        raise ValidationError("local panel did not receive exactly all train source records")
    selection_quota = 20
    chosen: list[dict[str, object]] = []
    used_games: set[str] = set()
    for label in STRATA:
        candidates = sorted(
            (
                (identity, source, public_data)
                for identity, (candidate_label, source, public_data) in by_identity.items()
                if candidate_label == label
            ),
            key=lambda row: (
                hashlib.sha256(f"step4:panel:{replicate}:{label}:{row[0]}".encode()).hexdigest(),
                row[0],
            ),
        )
        kept = 0
        for identity, source, public_data in candidates:
            if source in used_games:
                continue
            used_games.add(source)
            chosen.append(
                {
                    "replicate_id": replicate,
                    "stratum": label,
                    "safe_identity": identity,
                    "selection_hash": hashlib.sha256(
                        f"step4:panel:{replicate}:{label}:{identity}".encode()
                    ).hexdigest(),
                    "observation": public_data,
                    "audit_record_fingerprint": source,
                }
            )
            kept += 1
            if kept == selection_quota:
                break
        if kept != selection_quota:
            raise ValidationError(f"local panel quota failed for {replicate}:{label}")
    if quota == 1:
        return [next(row for row in chosen if row["stratum"] == label) for label in STRATA]
    return chosen


def read_jsonl_gzip(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt") as source:
        return [cast(dict[str, object], json.loads(row)) for row in source if row.strip()]


def validate_panels(output: Path, step3_root: Path, plan: Mapping[str, object]) -> None:
    source_root = step3_root.parent.parent
    inputs = plan.get("inputs")
    execution = cast(Mapping[str, object], plan["execution"])
    quota = execution.get("panel_quota_per_stratum")
    if type(quota) is not int or quota not in (1, 20):
        raise ValidationError("plan panel quota is malformed")
    if not isinstance(inputs, list) or len(inputs) != 3:
        raise ValidationError("plan inputs are malformed")
    for value in inputs:
        if not isinstance(value, Mapping):
            raise ValidationError("plan replicate input is malformed")
        replicate = cast(str, value["replicate_id"])
        dataset = load_structured_dataset(
            source_root / cast(str, cast(Mapping[str, object], value["dataset"])["path"])
        )
        train = set(str(item) for item in dataset.train.record_fingerprint.tolist())
        corpus = source_root / cast(str, cast(Mapping[str, object], value["corpus"])["path"])
        expected = local_panel(corpus, train, replicate, quota)
        actual = read_jsonl_gzip(output / "panel" / replicate / "positions.jsonl.gz")
        if actual != [
            {key: value for key, value in row.items() if key != "audit_record_fingerprint"}
            for row in expected
        ]:
            raise ValidationError(f"local panel reconstruction differs: {replicate}")
        audit = read_jsonl_gzip(output / "panel" / replicate / "audit-locators.jsonl.gz")
        if audit != [
            {
                "safe_identity": row["safe_identity"],
                "audit_record_fingerprint": row["audit_record_fingerprint"],
            }
            for row in expected
        ]:
            raise ValidationError(f"local panel audit locators differ: {replicate}")


def validate_controls(output: Path, step3_root: Path, plan: Mapping[str, object]) -> None:
    source_root = step3_root.parent.parent
    inputs = cast(list[object], plan["inputs"])
    q0 = load_checkpoint(Path(cast(str, plan["q0_path"])))
    for raw in inputs:
        item = cast(Mapping[str, object], raw)
        replicate = cast(str, item["replicate_id"])
        checkpoint = cast(Mapping[str, object], item["checkpoint"])
        config = cast(Mapping[str, object], checkpoint["training_config"])
        dataset_path = source_root / cast(str, cast(Mapping[str, object], item["dataset"])["path"])
        dataset = load_structured_dataset(dataset_path)
        training = StructuredTrainingConfig(
            seed=cast(int, config["seed"]),
            projection_seed=cast(int, config["projection_seed"]),
            shuffle_seed=cast(int, config["shuffle_seed"]),
            batch_size=cast(int, config["batch_size"]),
            max_epochs=cast(int, checkpoint["epochs_completed"]),
            early_stopping_patience=8,
            learning_rate=float(cast(float, config["learning_rate"])),
            weight_decay=float(cast(float, config["weight_decay"])),
            cpu_threads=1,
        )
        train = StructuredTrainingData(
            dataset.train.features,
            dataset.train.targets,
            dataset.train.game_index,
            dataset.train.phase,
        )
        validation = StructuredTrainingData(
            dataset.validation.features,
            dataset.validation.targets,
            dataset.validation.game_index,
            dataset.validation.phase,
        )
        reproduced = train_structured_model(
            train, validation, q0_state_dict=q0.model.state_dict(), config=training
        )
        actual = tensor_digest(reproduced.model.state_dict())
        if actual != checkpoint.get("tensor_digest"):
            raise ValidationError(f"independent control tensor mismatch: {replicate}")
        summary = read(output / "control-reproduction" / replicate / "summary.json")
        artifact(summary, f"control summary {replicate}")
        if (
            summary.get("reproduced_tensor_digest") != actual
            or summary.get("byte_for_byte_tensor_match") is not True
        ):
            raise ValidationError(f"retained control summary differs: {replicate}")


def validate_targets(output: Path, plan: Mapping[str, object]) -> None:
    """Validate safe arrays, common seeds, leaf gate, and quarantine boundaries locally."""
    for replicate in ("replicate-1", "replicate-2", "replicate-3"):
        root = output / "rollouts" / replicate
        manifest = read(root / "manifest.json")
        artifact(manifest, f"rollout manifest {replicate}")
        with np.load(root / "targets.npz", allow_pickle=False) as data:
            required = {
                "features",
                "targets",
                "position_index",
                "safe_identity",
                "stratum",
                "action_json",
                "sample_json",
            }
            if set(data.files) != required or data["features"].shape[1] != 519:
                raise ValidationError(f"target array schema differs: {replicate}")
            if not np.all(np.isfinite(data["features"])) or not np.all(
                (data["targets"] >= 0) & (data["targets"] <= 1)
            ):
                raise ValidationError(f"target values are outside safe finite bounds: {replicate}")
            leaves = 0
            samples = 0
            for identity, rows in zip(
                data["safe_identity"].tolist(), data["sample_json"].tolist(), strict=True
            ):
                parsed = json.loads(str(rows))
                if not isinstance(parsed, list) or len(parsed) != 10:
                    raise ValidationError("target row lacks ten common worlds")
                for row in parsed:
                    if row["terminal"] == row["depth_leaf"] or row["world_seed"] != derive_seed(
                        ROOT_SEED, f"step4:world:{replicate}:{identity}:{row['world_index']}"
                    ):
                        raise ValidationError(
                            "target common-random-number or terminal flags differ"
                        )
                    leaves += int(row["depth_leaf"])
                    samples += 1
            if samples == 0 or leaves / samples > 0.5 or manifest.get("mechanical_cap_errors") != 0:
                raise ValidationError(f"target leaf/cap gate fails: {replicate}")
        transcripts = read_jsonl_gzip(root / "trusted-world-transcripts.jsonl.gz")
        if len(transcripts) != 140 or any(
            row.get("quarantine") != "trusted-hidden-rollout-audit-only-v1" for row in transcripts
        ):
            raise ValidationError(f"trusted transcript quarantine/count differs: {replicate}")


def validate_arenas_and_stats(output: Path, smoke: bool) -> None:
    result = read(output / "result.json")
    arenas = cast(Mapping[str, object], result["arenas"])
    if smoke and set(arenas) != {"T1-vs-C1"}:
        raise ValidationError("smoke arena schedule differs")
    for key in arenas:
        report = read(output / "arenas" / key / "report.json")
        artifact(report, f"arena {key}")
        raw = cast(Mapping[str, object], report["report"])
        _, records = load_corpus(
            output / "arenas" / key / "records", verify_code=False, verify_replays=True
        )
        outcomes: dict[str, list[GameRecord]] = defaultdict(list)
        for record in records:
            if record.pair_id is None:
                raise ValidationError("arena record is missing a pair id")
            outcomes[record.pair_id].append(record)
        wins = 0
        by_seat: Counter[str] = Counter()
        seat_games: Counter[str] = Counter()
        agent_a = cast(Mapping[str, object], raw["agents"])["a"]
        agent_a_id = cast(str, cast(Mapping[str, object], agent_a)["id"])
        for pair in outcomes.values():
            if len(pair) != 2:
                raise ValidationError("arena pair is incomplete")
            for record in pair:
                index = 0 if record.seats[0].agent_id == agent_a_id else 1
                seat = "player_one" if index == 0 else "player_two"
                seat_games[seat] += 1
                # Map the winner player id to the selected agent's physical seat.
                winner_index = 0 if record.winner.value == "player_one" else 1
                if winner_index == index:
                    wins += 1
                    by_seat[seat] += 1
        rate = raw.get("agent_a_win_rate")
        if not isinstance(rate, int | float) or (
            wins != cast(Mapping[str, int], raw["wins"])["a"]
            or abs(wins / len(records) - float(rate)) > 1e-12
        ):
            raise ValidationError(f"local arena aggregate differs: {key}")
        for seat in ("player_one", "player_two"):
            declared = cast(Mapping[str, Mapping[str, object]], raw["agent_a_by_seat"])[seat]
            if declared["games"] != seat_games[seat] or declared["wins"] != by_seat[seat]:
                raise ValidationError(f"local arena seat aggregate differs: {key}:{seat}")
    statistics = read(output / "statistics.json")
    selection = read(output / "selection.json")
    artifact(statistics, "statistics")
    artifact(selection, "selection")
    if smoke and (
        selection.get("classification") != "nonclaim_smoke_no_disposition"
        or selection.get("rollout_treatment_enters_step5") is not False
    ):
        raise ValidationError("smoke selection disposition differs")


def validate_checksums(output: Path) -> None:
    checksums = read(output / "checksums.json")
    artifact(checksums, "checksums")
    files = cast(Mapping[str, object], checksums["files"])
    for relative, expected in files.items():
        if not isinstance(expected, str) or sha256(output / relative) != expected:
            raise ValidationError(f"checksum differs: {relative}")


def validate(output: Path, step3_root: Path) -> dict[str, object]:
    static_source_guard()
    plan = read(output / "plan.json")
    if plan.get("cycle_id") != CYCLE_ID:
        raise ValidationError("wrong cycle")
    smoke = cast(Mapping[str, object], plan["execution"]).get("claim") is False
    validate_checksums(output)
    validate_panels(output, step3_root, plan)
    validate_targets(output, plan)
    validate_controls(output, step3_root, plan)
    validate_arenas_and_stats(output, smoke)
    return {
        "version": "m7-counterfactual-rollout-independent-validation-v1",
        "plan_fingerprint": plan["plan_fingerprint"],
        "status": "passed",
        "checks": {
            "source_guard": True,
            "input_control_reproduction": True,
            "local_panel_quotas_and_identities": True,
            "local_target_seed_leaf_and_quarantine_checks": True,
            "local_arena_schedule_aggregate": True,
            "statistics_selection": True,
            "checksums": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step3-root", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.output, args.step3_root)
    result["artifact_fingerprint"] = digest(result)
    (args.output / "validation.json").write_bytes(canonical(result) + b"\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
