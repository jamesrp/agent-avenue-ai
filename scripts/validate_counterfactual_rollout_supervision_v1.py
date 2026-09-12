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
import math
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Any, cast

import numpy as np

from agent_avenue.agents import (
    Agent,
    GreedyHeuristicAgent,
    GreedyHeuristicConfig,
    RandomAgent,
    RandomAgentConfig,
    RandomSource,
    TerminalOffenseAgent,
    TerminalSafetyAgent,
    filter_immediate_win_actions,
    filter_terminal_actions,
)
from agent_avenue.agents.ordering import semantic_action_key
from agent_avenue.encoding.candidate_structured_v2 import encode_candidate
from agent_avenue.engine import apply_action, new_game
from agent_avenue.engine.cards import CANONICAL_DECK, CardName, recruit_effect
from agent_avenue.engine.model import (
    Action,
    CompletedTurn,
    OfferSlot,
    Phase,
    PlayerId,
    PlayOfferAction,
    RecruitAction,
)
from agent_avenue.engine.setup import normalize_config
from agent_avenue.learning import (
    StructuredTrainingConfig,
    StructuredTrainingData,
    inspect_structured_checkpoint,
    load_checkpoint,
    load_structured_dataset,
    tensor_digest,
    train_structured_model,
)
from agent_avenue.observation.model import (
    PlayContext,
    PlayerObservation,
    PublicPlayer,
    RecruitContext,
)
from agent_avenue.storage import (
    GameRecord,
    code_fingerprint,
    game_record_fingerprint,
    load_corpus,
    rules_fingerprint,
)

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
        "agent_avenue.observation.build",
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


def local_action_data(action: Action) -> dict[str, object]:
    if isinstance(action, RecruitAction):
        return {
            "type": "recruit",
            "revision": action.revision,
            "actor": action.actor.value,
            "slot": action.slot.value,
        }
    return {
        "type": "play_offer",
        "revision": action.revision,
        "actor": action.actor.value,
        "face_up": action.face_up.value,
        "face_down": action.face_down.value,
    }


def local_observation_data(observation: PlayerObservation) -> dict[str, object]:
    decision = observation.decision
    if isinstance(decision, PlayContext):
        decision_data: dict[str, object] = {
            "kind": "play",
            "revision": decision.revision,
            "actor": decision.actor.value,
        }
    elif isinstance(decision, RecruitContext):
        decision_data = {
            "kind": "recruit",
            "revision": decision.revision,
            "actor": decision.actor.value,
            "offered_by": decision.offered_by.value,
            "face_up": decision.face_up.value,
            "known_face_down": None
            if decision.known_face_down is None
            else decision.known_face_down.value,
            "slots": [slot.value for slot in decision.slots],
        }
    else:
        raise ValidationError("terminal observation cannot be a panel member")
    return {
        "viewer": observation.viewer.value,
        "own_hand": [card.value for card in observation.own_hand],
        "players": [
            {
                "player": player.player.value,
                "score": player.score,
                "recruited": [card.value for card in player.recruited],
                "hand_size": player.hand_size,
            }
            for player in observation.players
        ],
        "active_player": observation.active_player.value,
        "turn": observation.turn,
        "phase": observation.phase.value,
        "remaining_deck_count": observation.remaining_deck_count,
        "history": [
            {
                "turn": item.turn,
                "active_player": item.active_player.value,
                "face_up": item.face_up.value,
                "face_down": item.face_down.value,
                "chosen_slot": item.chosen_slot.value,
                "opponent_recruited": item.opponent_recruited.value,
                "active_recruited": item.active_recruited.value,
                "score_changes": list(item.score_changes),
            }
            for item in observation.history
        ],
        "decision": decision_data,
        "legal_actions": [local_action_data(action) for action in observation.legal_actions],
    }


def local_engine_observation(state: object, viewer: PlayerId) -> PlayerObservation:
    """Locally project a replayed authoritative state without calling production observe()."""
    raw: Any = state
    decision: PlayContext | RecruitContext
    if raw.phase is Phase.PLAY:
        decision = PlayContext("play", raw.revision, raw.active_player)
        actor = raw.active_player
    elif raw.phase is Phase.RECRUIT:
        if raw.offer is None:
            raise ValidationError("replayed recruit state has no offer")
        actor = raw.active_player.other()
        decision = RecruitContext(
            "recruit", raw.revision, actor, raw.active_player, raw.offer.face_up, None
        )
    else:
        raise ValidationError("replayed terminal state cannot enter panel")
    if viewer is not actor:
        raise ValidationError("local replay observer has the wrong viewer")
    if raw.phase is Phase.PLAY:
        hand = raw.hands[player_index_local(raw.active_player)]
        names = tuple(dict.fromkeys(hand))
        actions: tuple[Action, ...]
        if len(names) == 1:
            actions = (PlayOfferAction(raw.revision, raw.active_player, names[0], names[0]),)
        else:
            actions = tuple(
                PlayOfferAction(raw.revision, raw.active_player, up, down)
                for up in names
                for down in names
                if up is not down
            )
    else:
        actions = tuple(RecruitAction(raw.revision, actor, slot) for slot in OfferSlot)
    players = tuple(
        PublicPlayer(
            player,
            raw.scores[player_index_local(player)],
            raw.recruited[player_index_local(player)],
            len(raw.hands[player_index_local(player)]),
        )
        for player in PlayerId
    )
    return PlayerObservation(
        viewer,
        raw.hands[player_index_local(viewer)],
        (players[0], players[1]),
        raw.active_player,
        raw.turn,
        raw.phase,
        len(raw.deck),
        raw.history,
        decision,
        canonical_actions(actions),
    )


def safe_identity(observation: PlayerObservation) -> str:
    actions = canonical_actions(observation.legal_actions)
    decision = observation.decision
    actor = getattr(decision, "actor", None)
    revision = getattr(decision, "revision", None)
    if observation.viewer is not actor or type(revision) is not int:
        raise ValidationError("local panel observation is not the decision actor view")
    canonical_observation = local_observation_data(
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
            observation = local_engine_observation(state, actor)
            label = stratum(observation)
            if label is not None:
                canonical_data = local_observation_data(
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


@dataclass(frozen=True, slots=True)
class LocalOffer:
    offered_by: PlayerId
    face_up: CardName
    face_down: CardName


@dataclass(frozen=True, slots=True)
class LocalOutcome:
    winner: PlayerId
    reason: str
    resolution: str


@dataclass(frozen=True, slots=True)
class LocalLatentState:
    """Validator-owned synthetic state; never converted to an engine GameState."""

    deck: tuple[CardName, ...]
    hands: tuple[tuple[CardName, ...], tuple[CardName, ...]]
    recruited: tuple[tuple[CardName, ...], tuple[CardName, ...]]
    scores: tuple[int, int]
    active_player: PlayerId
    turn: int
    phase: Phase
    revision: int
    offer: LocalOffer | None = None
    history: tuple[CompletedTurn, ...] = ()
    outcome: LocalOutcome | None = None


EXPANDED_DECK = (
    *(CardName.DOUBLE_AGENT for _ in range(6)),
    *(CardName.ENFORCER for _ in range(6)),
    *(CardName.CODEBREAKER for _ in range(6)),
    *(CardName.DAREDEVIL for _ in range(6)),
    *(CardName.SABOTEUR for _ in range(6)),
    *(CardName.SENTINEL for _ in range(6)),
    CardName.SIDEKICK,
    CardName.MOLE,
)
POLICY_SLOTS = (
    ("q0-1", "q0"),
    ("q0-2", "q0"),
    ("q0-3", "q0"),
    ("q0-4", "q0"),
    ("q1", "q1"),
    ("q2", "q2"),
    ("q3", "q3"),
    ("q4", "q4"),
    ("greedy-public-v1", "heuristic"),
    ("random", "random"),
)


def derive_seed_local(root_seed: int, domain: str) -> int:
    if type(root_seed) is not int or not domain:
        raise ValidationError("local seed inputs are malformed")
    payload = canonical({"domain": domain, "root_seed": root_seed, "version": "sha256-domain-v1"})
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


@dataclass(slots=True)
class LocalRandom:
    seed: int
    domain: str
    counter: int = 0

    def randbelow(self, upper: int) -> int:
        if type(upper) is not int or upper <= 0:
            raise ValidationError("local RNG upper bound is malformed")
        modulus = 1 << 256
        limit = modulus - modulus % upper
        while True:
            payload = canonical(
                {
                    "algorithm": "sha256-counter-rejection-v1",
                    "counter": self.counter,
                    "domain": self.domain,
                    "seed": self.seed,
                }
            )
            self.counter += 1
            value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
            if value < limit:
                return value % upper


def card_index(card: CardName) -> int:
    return tuple(CardName).index(card)


def player_index_local(player: PlayerId) -> int:
    return 0 if player is PlayerId.PLAYER_ONE else 1


def canonical_cards(cards: Iterable[CardName]) -> tuple[CardName, ...]:
    return tuple(sorted(cards, key=card_index))


def local_adjudicate(
    *,
    scores: tuple[int, int],
    recruited: tuple[tuple[CardName, ...], tuple[CardName, ...]],
    active_player: PlayerId,
    deck_empty: bool,
    next_player_hand_size: int,
) -> LocalOutcome | None:
    players = (PlayerId.PLAYER_ONE, PlayerId.PLAYER_TWO)
    score_winners = tuple(
        player
        for player in players
        if scores[player_index_local(player)] >= scores[player_index_local(player.other())] + 7
    )
    counts = tuple(Counter(cards) for cards in recruited)
    instant_winners = tuple(
        player
        for player in players
        if counts[player_index_local(player)][CardName.CODEBREAKER] >= 3
    )
    instant_losers = tuple(
        player for player in players if counts[player_index_local(player)][CardName.DAREDEVIL] >= 3
    )
    candidates = set(score_winners) | set(instant_winners)
    candidates.update(player.other() for player in instant_losers)
    ordered = tuple(player for player in players if player in candidates)
    if ordered:
        return LocalOutcome(
            ordered[0] if len(ordered) == 1 else active_player,
            "condition",
            "sole_candidate" if len(ordered) == 1 else "active_condition_tie",
        )
    if deck_empty and next_player_hand_size < 2:
        next_player = active_player.other()
        active_score = scores[player_index_local(active_player)]
        next_score = scores[player_index_local(next_player)]
        if active_score == next_score:
            return LocalOutcome(active_player, "deck_exhaustion", "active_score_tie")
        return LocalOutcome(
            active_player if active_score > next_score else next_player,
            "deck_exhaustion",
            "high_score",
        )
    return None


def validate_local_state(state: LocalLatentState) -> None:
    if state.phase not in (Phase.PLAY, Phase.RECRUIT, Phase.TERMINAL):
        raise ValidationError("local state has invalid phase")
    if not 1 <= state.turn <= 19 or state.revision < 0 or len(state.hands) != 2:
        raise ValidationError("local state turn/revision/hand shape is invalid")
    if any(len(hand) > 4 for hand in state.hands):
        raise ValidationError("local state hand size exceeds four")
    cards = [
        *state.deck,
        *state.hands[0],
        *state.hands[1],
        *state.recruited[0],
        *state.recruited[1],
    ]
    if state.offer is not None:
        cards.extend((state.offer.face_up, state.offer.face_down))
    if Counter(cards) != Counter(EXPANDED_DECK):
        raise ValidationError("local latent card conservation failed")
    if state.phase is Phase.PLAY and (state.offer is not None or state.outcome is not None):
        raise ValidationError("local play state has offer/outcome")
    if state.phase is Phase.RECRUIT and (state.offer is None or state.outcome is not None):
        raise ValidationError("local recruit state is malformed")
    if state.phase is Phase.TERMINAL and (state.offer is not None or state.outcome is None):
        raise ValidationError("local terminal state is malformed")
    expected_revision = len(state.history) * 2 + (1 if state.phase is Phase.RECRUIT else 0)
    if expected_revision != state.revision:
        raise ValidationError("local state revision is inconsistent")


def local_legal_actions(state: LocalLatentState) -> tuple[Action, ...]:
    validate_local_state(state)
    if state.phase is Phase.TERMINAL:
        return ()
    if state.phase is Phase.PLAY:
        hand = state.hands[player_index_local(state.active_player)]
        names = tuple(dict.fromkeys(hand))
        if len(hand) < 2:
            return ()
        if len(names) == 1:
            return (PlayOfferAction(state.revision, state.active_player, names[0], names[0]),)
        return tuple(
            PlayOfferAction(state.revision, state.active_player, up, down)
            for up in names
            for down in names
            if up is not down
        )
    return tuple(
        RecruitAction(state.revision, state.active_player.other(), slot) for slot in OfferSlot
    )


def local_decision_actor(state: LocalLatentState) -> PlayerId:
    if state.phase is Phase.TERMINAL:
        raise ValidationError("terminal local state has no decision actor")
    return state.active_player if state.phase is Phase.PLAY else state.active_player.other()


def local_observation(state: LocalLatentState, viewer: PlayerId) -> PlayerObservation:
    validate_local_state(state)
    legal = canonical_actions(local_legal_actions(state))
    decision: PlayContext | RecruitContext
    if state.phase is Phase.PLAY:
        decision = PlayContext("play", state.revision, state.active_player)
    else:
        assert state.offer is not None
        decision = RecruitContext(
            "recruit",
            state.revision,
            state.active_player.other(),
            state.offer.offered_by,
            state.offer.face_up,
            None,
        )
    players = tuple(
        PublicPlayer(
            player,
            state.scores[player_index_local(player)],
            state.recruited[player_index_local(player)],
            len(state.hands[player_index_local(player)]),
        )
        for player in PlayerId
    )
    return PlayerObservation(
        viewer,
        canonical_cards(state.hands[player_index_local(viewer)]),
        (players[0], players[1]),
        state.active_player,
        state.turn,
        state.phase,
        len(state.deck),
        state.history,
        decision,
        legal,
    )


def parse_action(value: object) -> Action:
    if not isinstance(value, Mapping):
        raise ValidationError("target action is malformed")
    try:
        revision = value["revision"]
        actor = PlayerId(cast(str, value["actor"]))
        action_type = value["type"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError("target action fields are malformed") from exc
    if type(revision) is not int:
        raise ValidationError("target action revision is malformed")
    if action_type == "recruit":
        return RecruitAction(revision, actor, OfferSlot(cast(str, value["slot"])))
    if action_type == "play_offer":
        return PlayOfferAction(
            revision,
            actor,
            CardName(cast(str, value["face_up"])),
            CardName(cast(str, value["face_down"])),
        )
    raise ValidationError("target action type is unknown")


def parse_observation(value: object) -> PlayerObservation:
    if not isinstance(value, Mapping):
        raise ValidationError("panel observation is malformed")
    try:
        history = tuple(
            CompletedTurn(
                cast(int, item["turn"]),
                PlayerId(cast(str, item["active_player"])),
                CardName(cast(str, item["face_up"])),
                CardName(cast(str, item["face_down"])),
                OfferSlot(cast(str, item["chosen_slot"])),
                CardName(cast(str, item["opponent_recruited"])),
                CardName(cast(str, item["active_recruited"])),
                (
                    cast(int, cast(list[object], item["score_changes"])[0]),
                    cast(int, cast(list[object], item["score_changes"])[1]),
                ),
            )
            for item in cast(list[Mapping[str, object]], value["history"])
        )
        players = tuple(
            PublicPlayer(
                PlayerId(cast(str, item["player"])),
                cast(int, item["score"]),
                tuple(CardName(cast(str, card)) for card in cast(list[object], item["recruited"])),
                cast(int, item["hand_size"]),
            )
            for item in cast(list[Mapping[str, object]], value["players"])
        )
        if len(players) != 2:
            raise ValidationError("panel observation player count is invalid")
        decision_data = cast(Mapping[str, object], value["decision"])
        decision: PlayContext | RecruitContext
        if decision_data["kind"] == "play":
            decision = PlayContext(
                "play",
                cast(int, decision_data["revision"]),
                PlayerId(cast(str, decision_data["actor"])),
            )
        elif decision_data["kind"] == "recruit":
            known = decision_data.get("known_face_down")
            decision = RecruitContext(
                "recruit",
                cast(int, decision_data["revision"]),
                PlayerId(cast(str, decision_data["actor"])),
                PlayerId(cast(str, decision_data["offered_by"])),
                CardName(cast(str, decision_data["face_up"])),
                None if known is None else CardName(cast(str, known)),
                (
                    OfferSlot(cast(str, cast(list[object], decision_data["slots"])[0])),
                    OfferSlot(cast(str, cast(list[object], decision_data["slots"])[1])),
                ),
            )
        else:
            raise ValidationError("panel terminal observation is not eligible")
        actions = tuple(parse_action(item) for item in cast(list[object], value["legal_actions"]))
        return PlayerObservation(
            PlayerId(cast(str, value["viewer"])),
            tuple(CardName(cast(str, card)) for card in cast(list[object], value["own_hand"])),
            (players[0], players[1]),
            PlayerId(cast(str, value["active_player"])),
            cast(int, value["turn"]),
            Phase(cast(str, value["phase"])),
            cast(int, value["remaining_deck_count"]),
            history,
            decision,
            canonical_actions(actions),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError("panel observation cannot be locally reconstructed") from exc


def local_sample_world(observation: PlayerObservation, rng: LocalRandom) -> LocalLatentState:
    if observation.viewer is not getattr(observation.decision, "actor", None):
        raise ValidationError("sampled observation is not the decision actor view")
    if observation.phase not in (Phase.PLAY, Phase.RECRUIT):
        raise ValidationError("sampled observation phase is invalid")
    remaining = list(EXPANDED_DECK)
    for player in observation.players:
        for card in player.recruited:
            remaining.remove(card)
    own_hand = canonical_cards(observation.own_hand)
    for card in own_hand:
        remaining.remove(card)
    actor = observation.viewer
    opponent = actor.other()
    if observation.phase is Phase.PLAY:
        opponent_hand = tuple(
            remaining.pop(rng.randbelow(len(remaining)))
            for _ in range(observation.players[player_index_local(opponent)].hand_size)
        )
        for upper in range(len(remaining) - 1, 0, -1):
            lower = rng.randbelow(upper + 1)
            remaining[upper], remaining[lower] = remaining[lower], remaining[upper]
        hands = (
            (own_hand, opponent_hand) if actor is PlayerId.PLAYER_ONE else (opponent_hand, own_hand)
        )
        play_decision = observation.decision
        if not isinstance(play_decision, PlayContext):
            raise ValidationError("play sampler decision is malformed")
        state = LocalLatentState(
            tuple(remaining),
            hands,
            (observation.players[0].recruited, observation.players[1].recruited),
            (observation.players[0].score, observation.players[1].score),
            observation.active_player,
            observation.turn,
            Phase.PLAY,
            play_decision.revision,
            None,
            observation.history,
        )
    else:
        decision = observation.decision
        if not isinstance(decision, RecruitContext) or decision.known_face_down is not None:
            raise ValidationError("recruit sampler received an unsafe face-down value")
        remaining.remove(decision.face_up)
        face_down = remaining.pop(rng.randbelow(len(remaining)))
        offerer_hand = tuple(
            remaining.pop(rng.randbelow(len(remaining)))
            for _ in range(observation.players[player_index_local(decision.offered_by)].hand_size)
        )
        for upper in range(len(remaining) - 1, 0, -1):
            lower = rng.randbelow(upper + 1)
            remaining[upper], remaining[lower] = remaining[lower], remaining[upper]
        hands = (
            (own_hand, offerer_hand) if actor is PlayerId.PLAYER_ONE else (offerer_hand, own_hand)
        )
        state = LocalLatentState(
            tuple(remaining),
            hands,
            (observation.players[0].recruited, observation.players[1].recruited),
            (observation.players[0].score, observation.players[1].score),
            decision.offered_by,
            observation.turn,
            Phase.RECRUIT,
            decision.revision,
            LocalOffer(decision.offered_by, decision.face_up, face_down),
            observation.history,
        )
    if len(state.deck) != observation.remaining_deck_count:
        raise ValidationError("local sampler residual deck count differs from public observation")
    validate_local_state(state)
    return state


def local_transition(state: LocalLatentState, action: Action) -> LocalLatentState:
    validate_local_state(state)
    if state.phase is Phase.TERMINAL or action not in local_legal_actions(state):
        raise ValidationError("local transition action is illegal")
    if action.actor is not local_decision_actor(state) or action.revision != state.revision:
        raise ValidationError("local transition action ownership differs")
    if isinstance(action, PlayOfferAction):
        active_index = player_index_local(state.active_player)
        hand = list(state.hands[active_index])
        try:
            hand.remove(action.face_up)
            hand.remove(action.face_down)
        except ValueError as exc:
            raise ValidationError("local transition offered card is absent") from exc
        draw_count = min(4 - len(hand), len(state.deck))
        hands = list(state.hands)
        hands[active_index] = tuple(hand) + state.deck[:draw_count]
        next_state = LocalLatentState(
            state.deck[draw_count:],
            (hands[0], hands[1]),
            state.recruited,
            state.scores,
            state.active_player,
            state.turn,
            Phase.RECRUIT,
            state.revision + 1,
            LocalOffer(state.active_player, action.face_up, action.face_down),
            state.history,
        )
        validate_local_state(next_state)
        return next_state
    assert state.offer is not None
    active, opponent = state.active_player, state.active_player.other()
    if action.slot is OfferSlot.FACE_UP:
        opponent_card, active_card = state.offer.face_up, state.offer.face_down
    else:
        opponent_card, active_card = state.offer.face_down, state.offer.face_up
    active_index, opponent_index = player_index_local(active), player_index_local(opponent)
    recruited = [list(state.recruited[0]), list(state.recruited[1])]
    recruited[opponent_index].append(opponent_card)
    recruited[active_index].append(active_card)
    score_changes = [0, 0]
    for index, card in ((opponent_index, opponent_card), (active_index, active_card)):
        effect = recruit_effect(card, recruited[index].count(card))
        if effect.kind == "score":
            score_changes[index] = effect.points
    scores = (state.scores[0] + score_changes[0], state.scores[1] + score_changes[1])
    completed = CompletedTurn(
        state.turn,
        active,
        state.offer.face_up,
        state.offer.face_down,
        action.slot,
        opponent_card,
        active_card,
        (score_changes[0], score_changes[1]),
    )
    outcome = local_adjudicate(
        scores=scores,
        recruited=(tuple(recruited[0]), tuple(recruited[1])),
        active_player=active,
        deck_empty=not state.deck,
        next_player_hand_size=len(state.hands[opponent_index]),
    )
    if outcome is not None:
        result = LocalLatentState(
            state.deck,
            state.hands,
            (tuple(recruited[0]), tuple(recruited[1])),
            scores,
            active,
            state.turn,
            Phase.TERMINAL,
            state.revision + 1,
            None,
            (*state.history, completed),
            outcome,
        )
    else:
        if state.turn >= 19:
            raise ValidationError("local mechanical turn cap was reached")
        result = LocalLatentState(
            state.deck,
            state.hands,
            (tuple(recruited[0]), tuple(recruited[1])),
            scores,
            opponent,
            state.turn + 1,
            Phase.PLAY,
            state.revision + 1,
            None,
            (*state.history, completed),
            None,
        )
    validate_local_state(result)
    return result


def local_state_digest(state: LocalLatentState) -> str:
    offer = (
        None
        if state.offer is None
        else {
            "offered_by": state.offer.offered_by.value,
            "face_up": state.offer.face_up.value,
            "face_down": state.offer.face_down.value,
        }
    )
    return digest(
        {
            "deck": [card.value for card in state.deck],
            "hands": [[card.value for card in hand] for hand in state.hands],
            "recruited": [[card.value for card in cards] for cards in state.recruited],
            "scores": list(state.scores),
            "active_player": state.active_player.value,
            "turn": state.turn,
            "phase": state.phase.value,
            "revision": state.revision,
            "offer": offer,
        }
    )


def local_state_data(state: LocalLatentState) -> dict[str, object]:
    return {
        "deck": [card.value for card in state.deck],
        "hands": [[card.value for card in hand] for hand in state.hands],
        "recruited": [[card.value for card in cards] for cards in state.recruited],
        "scores": list(state.scores),
        "active_player": state.active_player.value,
        "turn": state.turn,
        "phase": state.phase.value,
        "revision": state.revision,
    }


def local_permutation(values: tuple[str, ...], rng: LocalRandom) -> tuple[str, ...]:
    result = list(values)
    for upper in range(len(result) - 1, 0, -1):
        lower = rng.randbelow(upper + 1)
        result[upper], result[lower] = result[lower], result[upper]
    return tuple(result)


def local_assignments(replicate: str, identity: str) -> tuple[tuple[str, str], ...]:
    slots = tuple(slot for slot, _ in POLICY_SLOTS)
    p1 = local_permutation(
        slots,
        LocalRandom(
            derive_seed_local(
                ROOT_SEED, f"step4:policy-permutation:{replicate}:{identity}:player_one"
            ),
            "step4:policy-permutation/p1",
        ),
    )
    p2 = local_permutation(
        slots,
        LocalRandom(
            derive_seed_local(
                ROOT_SEED, f"step4:policy-permutation:{replicate}:{identity}:player_two"
            ),
            "step4:policy-permutation/p2",
        ),
    )
    return tuple((p1[index], p2[index]) for index in range(10))


def policy_id(slot: str) -> str:
    return dict(POLICY_SLOTS)[slot]


def policy_bundle(plan: Mapping[str, object]) -> tuple[dict[str, object], object]:
    from agent_avenue.agents.learned import LearnedValueAgent

    paths = cast(Mapping[str, object], plan["opponent_paths"])

    def learned(name: str) -> object:
        return LearnedValueAgent.from_checkpoint(load_checkpoint(Path(cast(str, paths[name]))))

    q0 = learned("q0-parent")
    return (
        {
            "q0": q0,
            "q1": learned("q1"),
            "q2": learned("q2"),
            "q3": learned("q3"),
            "q4": learned("q4"),
            "heuristic": GreedyHeuristicAgent(GreedyHeuristicConfig()),
            "random": RandomAgent(RandomAgentConfig()),
        },
        q0,
    )


@dataclass(slots=True)
class _CaptureAgent:
    actions: tuple[Action, ...] = ()

    def choose_action(
        self,
        observation: PlayerObservation,
        decision: PlayContext | RecruitContext | object,
        legal_actions: tuple[Action, ...],
        rng: RandomSource,
    ) -> Action:
        del observation, decision, rng
        self.actions = legal_actions
        return legal_actions[0]


def enveloped_actions_local(observation: PlayerObservation) -> tuple[Action, ...]:
    """Obtain the frozen offense+safety legal set through a capture policy on a local observer."""
    capture = _CaptureAgent()
    wrapper = TerminalOffenseAgent(TerminalSafetyAgent(cast(Agent, capture)))
    wrapper.choose_action(
        observation,
        observation.decision,
        observation.legal_actions,
        LocalRandom(0, "validator-envelope"),
    )
    return canonical_actions(capture.actions)


def local_rollout(
    observation: PlayerObservation,
    replicate: str,
    identity: str,
    candidate: Action,
    world_index: int,
    policies: Mapping[str, object],
    leaf: object,
    *,
    transcript: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    seed = derive_seed_local(ROOT_SEED, f"step4:world:{replicate}:{identity}:{world_index}")
    state = local_sample_world(
        observation, LocalRandom(seed, f"step4:world/{identity}/{world_index}")
    )
    assignments = local_assignments(replicate, identity)
    p1_slot, p2_slot = assignments[world_index]
    root_actor = observation.viewer
    steps: list[dict[str, object]] = []
    if transcript:
        steps.append(
            {
                "ordinal": 0,
                "forced_root": True,
                "action": local_action_data(candidate),
                "state_before": local_state_data(state),
            }
        )
    state = local_transition(state, candidate)
    if transcript:
        steps[-1]["state_after"] = local_state_data(state)
    rng_seeds: list[int] = []
    applied = 1
    while state.phase is not Phase.TERMINAL and applied < 9:
        actor = local_decision_actor(state)
        identifier = policy_id(p1_slot if actor is PlayerId.PLAYER_ONE else p2_slot)
        base: Agent = cast(Agent, policies[identifier])
        wrapped = TerminalOffenseAgent(TerminalSafetyAgent(base))
        local_view = local_observation(state, actor)
        decision_seed = derive_seed_local(
            ROOT_SEED,
            f"step4:rollout-rng:{replicate}:{identity}:{world_index}:{actor.value}:{state.revision}",
        )
        policy_rng = LocalRandom(
            decision_seed, f"step4:rollout-rng/{world_index}/{actor.value}/{state.revision}"
        )
        action = wrapped.choose_action(
            local_view, local_view.decision, local_view.legal_actions, policy_rng
        )
        if action not in local_view.legal_actions:
            raise ValidationError("local continuation policy escaped its safe legal set")
        rng_seeds.append(decision_seed)
        if transcript:
            steps.append(
                {
                    "ordinal": applied,
                    "forced_root": False,
                    "rng_seed": decision_seed,
                    "action": local_action_data(action),
                    "state_before": local_state_data(state),
                }
            )
        state = local_transition(state, action)
        if transcript:
            steps[-1]["state_after"] = local_state_data(state)
        applied += 1
    if state.phase is Phase.TERMINAL:
        assert state.outcome is not None
        root_value = 1.0 if state.outcome.winner is root_actor else 0.0
        terminal, depth_leaf = True, False
    else:
        if state.turn > 19:
            raise ValidationError("local depth leaf exceeded the mechanical turn cap")
        leaf_actor = local_decision_actor(state)
        leaf_view = local_observation(state, leaf_actor)
        allowed = enveloped_actions_local(leaf_view)
        leaf_view = PlayerObservation(
            leaf_view.viewer,
            leaf_view.own_hand,
            leaf_view.players,
            leaf_view.active_player,
            leaf_view.turn,
            leaf_view.phase,
            leaf_view.remaining_deck_count,
            leaf_view.history,
            leaf_view.decision,
            allowed,
        )
        leaf_agent: Any = leaf
        scored: Any = leaf_agent.score_candidates(leaf_view, leaf_view.decision, allowed)
        logits = tuple(float(value) for value in scored.logits)
        leaf_value = (
            1.0 / (1.0 + exp(-max(logits)))
            if max(logits) >= 0
            else exp(max(logits)) / (1.0 + exp(max(logits)))
        )
        root_value = leaf_value if leaf_actor is root_actor else 1.0 - leaf_value
        terminal, depth_leaf = False, True
    sample = {
        "world_index": world_index,
        "world_seed": seed,
        "latent_digest": local_state_digest(
            local_sample_world(
                observation, LocalRandom(seed, f"step4:world/{identity}/{world_index}")
            )
        ),
        "player_one_policy_id": policy_id(p1_slot),
        "player_two_policy_id": policy_id(p2_slot),
        "continuation_rng_seeds": rng_seeds,
        "terminal": terminal,
        "depth_leaf": depth_leaf,
        "actions_applied": applied,
        "root_value": root_value,
    }
    return sample, steps


def _sample_equal(expected: Mapping[str, object], actual: Mapping[str, object]) -> bool:
    if set(expected) != set(actual):
        return False
    for key, value in expected.items():
        if key == "root_value":
            actual_value = actual[key]
            if not isinstance(actual_value, int | float) or actual_value != cast(float, value):
                return False
        elif actual[key] != value:
            return False
    return True


def _panel_rows_for_target(output: Path, replicate: str) -> tuple[dict[str, object], ...]:
    rows = read_jsonl_gzip(output / "panel" / replicate / "positions.jsonl.gz")
    if not rows:
        raise ValidationError("target panel artifact is empty")
    return tuple(rows)


def validate_targets(output: Path, plan: Mapping[str, object]) -> dict[str, object]:
    """Fully regenerate every retained world/candidate target using validator-local mechanics."""
    started = time.perf_counter()
    policies, leaf = policy_bundle(plan)
    total_units = 0
    total_leaves = 0
    total_samples = 0
    cap_errors = 0
    transcript_count = 0
    for replicate in ("replicate-1", "replicate-2", "replicate-3"):
        root = output / "rollouts" / replicate
        manifest = read(root / "manifest.json")
        artifact(manifest, f"rollout manifest {replicate}")
        panel_rows = _panel_rows_for_target(output, replicate)
        observations: dict[str, PlayerObservation] = {}
        panel_by_index: dict[int, dict[str, object]] = {}
        for index, row in enumerate(panel_rows):
            if row.get("replicate_id") != replicate or row.get("stratum") not in STRATA:
                raise ValidationError("retained target panel row identity is malformed")
            observation = parse_observation(row.get("observation"))
            if safe_identity(observation) != row.get("safe_identity"):
                raise ValidationError("retained target panel safe identity differs")
            observations[cast(str, row["safe_identity"])] = observation
            panel_by_index[index] = row
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
            expected_panel = cast(list[object], read(root / "targets.json")["panel"])
            if len(expected_panel) != len(panel_rows):
                raise ValidationError("target manifest panel count differs")
            cursor_by_position: Counter[int] = Counter()
            for row_index in range(int(data["targets"].size)):
                position_index = int(data["position_index"][row_index])
                panel = panel_by_index.get(position_index)
                if panel is None:
                    raise ValidationError("target row position index is out of panel bounds")
                identity = str(data["safe_identity"][row_index])
                label = str(data["stratum"][row_index])
                if identity != panel["safe_identity"] or label != panel["stratum"]:
                    raise ValidationError("target row panel identity/order differs")
                observation = observations[identity]
                candidate = parse_action(json.loads(str(data["action_json"][row_index])))
                expected_actions = canonical_actions(observation.legal_actions)
                expected_candidate = expected_actions[cursor_by_position[position_index]]
                cursor_by_position[position_index] += 1
                if candidate != expected_candidate:
                    raise ValidationError("candidate semantic ordering differs")
                vector = np.asarray(
                    encode_candidate(observation, candidate).vector, dtype=np.float32
                )
                if (
                    vector.tobytes()
                    != np.asarray(data["features"][row_index], dtype=np.float32).tobytes()
                ):
                    raise ValidationError("target feature vector/digest differs")
                retained_samples = json.loads(str(data["sample_json"][row_index]))
                if not isinstance(retained_samples, list) or len(retained_samples) != 10:
                    raise ValidationError("target row lacks fixed ten worlds")
                expected_samples: list[dict[str, object]] = []
                for world_index in range(10):
                    try:
                        sample, _ = local_rollout(
                            observation,
                            replicate,
                            identity,
                            candidate,
                            world_index,
                            policies,
                            leaf,
                        )
                    except ValidationError:
                        cap_errors += 1
                        raise
                    retained = retained_samples[world_index]
                    if not isinstance(retained, Mapping) or not _sample_equal(sample, retained):
                        raise ValidationError("local rollout sample differs from retained target")
                    expected_samples.append(sample)
                mean = sum(cast(float, sample["root_value"]) for sample in expected_samples) / 10
                if (
                    np.asarray(mean, dtype=np.float32).tobytes()
                    != np.asarray(data["targets"][row_index], dtype=np.float32).tobytes()
                ):
                    raise ValidationError("ten-world rollout target differs")
                total_units += 10
                total_samples += 10
                total_leaves += sum(bool(sample["depth_leaf"]) for sample in expected_samples)
            for position_index, panel in panel_by_index.items():
                if cursor_by_position[position_index] != len(
                    observations[cast(str, panel["safe_identity"])].legal_actions
                ):
                    raise ValidationError(
                        "target rows do not cover every legal candidate exactly once"
                    )
        actual_transcripts = read_jsonl_gzip(root / "trusted-world-transcripts.jsonl.gz")
        expected_transcripts: list[dict[str, object]] = []
        for label in STRATA:
            panel = next(row for row in panel_rows if row["stratum"] == label)
            identity = cast(str, panel["safe_identity"])
            observation = observations[identity]
            candidate = canonical_actions(observation.legal_actions)[0]
            assignments = local_assignments(replicate, identity)
            for world_index in range(10):
                sample, steps = local_rollout(
                    observation,
                    replicate,
                    identity,
                    candidate,
                    world_index,
                    policies,
                    leaf,
                    transcript=True,
                )
                p1_slot, p2_slot = assignments[world_index]
                expected_transcripts.append(
                    {
                        "replicate_id": replicate,
                        "stratum": label,
                        "safe_identity": identity,
                        "candidate": local_action_data(candidate),
                        "world_index": world_index,
                        "world_seed": sample["world_seed"],
                        "player_one_policy_id": policy_id(p1_slot),
                        "player_two_policy_id": policy_id(p2_slot),
                        "steps": steps,
                        "terminal": sample["terminal"],
                        "quarantine": "trusted-hidden-rollout-audit-only-v1",
                    }
                )
        if actual_transcripts != expected_transcripts:
            raise ValidationError("local trusted transcript reconstruction differs")
        if len(actual_transcripts) != 140:
            raise ValidationError("trusted transcript count differs per replicate")
        transcript_count += len(actual_transcripts)
        declared = cast(Mapping[str, object], manifest)
        if declared.get("mechanical_cap_errors") != 0:
            raise ValidationError("retained rollout manifest records cap errors")
    elapsed = time.perf_counter() - started
    if transcript_count != 420 or total_samples == 0:
        raise ValidationError("trusted transcript/global target cardinality differs")
    leaf_fraction = total_leaves / total_samples
    if leaf_fraction > 0.5 or cap_errors != 0:
        raise ValidationError("local rollout leaf/cap eligibility gate fails")
    return {
        "target_work_units": total_units,
        "target_elapsed_seconds": elapsed,
        "target_units_per_second": total_units / elapsed if elapsed else float("inf"),
        "depth_leaf_fraction": leaf_fraction,
        "mechanical_cap_errors": cap_errors,
        "transcript_count": transcript_count,
    }


def local_wilson(wins: int, games: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = wins / games
    z2 = z * z
    denominator = 1 + z2 / games
    center = (proportion + z2 / (2 * games)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / games + z2 / (4 * games * games))
        / denominator
    )
    return (max(0.0, center - radius), min(1.0, center + radius))


def local_paired_bootstrap(pair_wins: tuple[int, ...], master_seed: int) -> dict[str, object]:
    seed = derive_seed_local(master_seed, "arena:paired-bootstrap:agent-a-win-rate:v1")
    rng = LocalRandom(seed, "arena:paired-bootstrap:agent-a-win-rate:v1")
    count = len(pair_wins)
    totals = [sum(pair_wins[rng.randbelow(count)] for _ in range(count)) for _ in range(20_000)]
    totals.sort()
    return {
        "method": "paired-percentile-bootstrap-v1",
        "statistic": "agent_a_win_rate",
        "unit": "two-game paired seed block",
        "confidence_level": 0.95,
        "point_estimate": sum(pair_wins) / (2 * count),
        "pair_count": count,
        "resample_count": 20_000,
        "order_statistic_indices": {"lower": 499, "upper": 19_499},
        "interval": [totals[499] / (2 * count), totals[19_499] / (2 * count)],
        "bootstrap_seed": seed,
        "bootstrap_rng_domain": "arena:paired-bootstrap:agent-a-win-rate:v1",
        "rng_algorithm": "sha256-counter-rejection-v1",
        "seed_derivation": "sha256-domain-v1",
    }


def local_cell_lanes(cell: Mapping[str, object]) -> tuple[str, str, str, str]:
    replicate = cast(str, cell["replicate_id"])
    candidate = cell.get("candidate")
    opponent = cast(str, cell["opponent"])
    shared = cast(str | None, cell.get("shared_group"))
    if candidate is None:
        return (
            "q0-parent",
            "heuristic",
            f"step4:{replicate}:heuristic:candidate",
            f"step4:{replicate}:heuristic:opponent",
        )
    candidate_id = f"{candidate}-{replicate}"
    candidate_rng = f"step4:{replicate}:{shared}:candidate"
    if opponent == "C":
        return candidate_id, f"C-{replicate}", candidate_rng, f"step4:{replicate}:direct:control"
    return candidate_id, opponent, candidate_rng, f"step4:{replicate}:{opponent}:opponent"


def local_expected_master_seed(cell: Mapping[str, object]) -> int:
    key = cast(str, cell["key"])
    replicate = cast(str, cell["replicate_id"])
    shared = cell.get("shared_group")
    domain = shared if isinstance(shared, str) else key
    return derive_seed_local(ROOT_SEED, f"step4:arena:{domain}:{replicate}") & ((1 << 63) - 1)


def local_validate_arena(
    directory: Path, artifact_value: Mapping[str, object], cell: Mapping[str, object]
) -> Mapping[str, object]:
    raw = cast(Mapping[str, object], artifact_value["report"])
    master = local_expected_master_seed(cell)
    if cell.get("master_seed") != master or raw.get("master_seed") != master:
        raise ValidationError("arena master seed does not use shared_group-or-key derivation")
    a_id, b_id, a_rng, b_rng = local_cell_lanes(cell)
    agents = cast(Mapping[str, Mapping[str, object]], raw["agents"])
    if agents["a"].get("id") != a_id or agents["b"].get("id") != b_id:
        raise ValidationError("arena candidate/opponent identity differs from frozen cell")
    pairs = cast(int, cell["paired_blocks"])
    _, records = load_corpus(directory / "records", verify_code=False, verify_replays=True)
    if len(records) != pairs * 2 or raw.get("total_games") != pairs * 2:
        raise ValidationError("arena record count differs from schedule")
    pair_wins: Counter[str] = Counter()
    pair_games: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    terminal: Counter[str] = Counter()
    seat_games: Counter[str] = Counter()
    seat_wins: Counter[str] = Counter()
    margin = turns = decisions = 0
    for pair_index in range(pairs):
        pair_id = f"pair-{pair_index:06d}"
        setup = derive_seed_local(master, f"arena:pair:{pair_index}:setup") & ((1 << 64) - 1)
        seed_a = derive_seed_local(master, f"arena:pair:{pair_index}:agent:{a_rng}")
        seed_b = derive_seed_local(master, f"arena:pair:{pair_index}:agent:{b_rng}")
        first, second = records[2 * pair_index], records[2 * pair_index + 1]
        for record, suffix, expected_ids, expected_seeds, expected_rngs in (
            (first, "a-first", (a_id, b_id), (seed_a, seed_b), (a_rng, b_rng)),
            (second, "b-first", (b_id, a_id), (seed_b, seed_a), (b_rng, a_rng)),
        ):
            if (
                record.run_id != cell["run_id"]
                or record.game_id != f"{pair_id}-{suffix}"
                or record.pair_id != pair_id
                or record.replay.seed != setup
                or tuple(seat.agent_id for seat in record.seats) != expected_ids
                or tuple(seat.seed for seat in record.seats) != expected_seeds
                or tuple(seat.rng_domain for seat in record.seats)
                != tuple(f"agent:{item}" for item in expected_rngs)
            ):
                raise ValidationError("arena record schedule/seat/RNG alignment differs")
            if (
                dict(record.seats[0].config)
                != agents["a" if expected_ids[0] == a_id else "b"]["config"]
            ):
                raise ValidationError("arena record seat configuration differs from report")
            winner_id = record.seats[0 if record.winner is PlayerId.PLAYER_ONE else 1].agent_id
            wins[winner_id] += 1
            pair_games[pair_id] += 1
            if winner_id == a_id:
                pair_wins[pair_id] += 1
            terminal[record.terminal_reason] += 1
            a_seat = "player_one" if record.seats[0].agent_id == a_id else "player_two"
            seat_games[a_seat] += 1
            if winner_id == a_id:
                seat_wins[a_seat] += 1
            index = 0 if a_seat == "player_one" else 1
            margin += record.final_scores[index] - record.final_scores[1 - index]
            turns += record.turn_count
            decisions += record.decision_count
    outcomes = [
        {"pair_id": pair_id, "agent_a_wins": pair_wins[pair_id]} for pair_id in sorted(pair_games)
    ]
    if any(pair_games[pair_id] != 2 for pair_id in pair_games):
        raise ValidationError("arena paired block is incomplete")
    a_wins = wins[a_id]
    expected_bootstrap = local_paired_bootstrap(
        tuple(pair_wins[item] for item in sorted(pair_games)), master
    )
    expected = {
        "run_id": cell["run_id"],
        "total_games": pairs * 2,
        "paired_seed_count": pairs,
        "wins": {"a": a_wins, "b": wins[b_id]},
        "agent_a_win_rate": a_wins / (pairs * 2),
        "confidence_interval_95": list(local_wilson(a_wins, pairs * 2)),
        "wilson_confidence_interval_95": list(local_wilson(a_wins, pairs * 2)),
        "paired_seed_outcomes": outcomes,
        "paired_bootstrap_confidence_interval_95": expected_bootstrap,
        "agent_a_by_seat": {
            seat: {
                "games": seat_games[seat],
                "wins": seat_wins[seat],
                "win_rate": seat_wins[seat] / seat_games[seat],
            }
            for seat in ("player_one", "player_two")
        },
        "terminal_reasons": dict(sorted(terminal.items())),
        "average_score_margin": margin / (pairs * 2),
        "average_turns": turns / (pairs * 2),
        "average_decisions": decisions / (pairs * 2),
        "master_seed": master,
        "seed_derivation": "sha256-domain-v1",
        "rng_algorithm": "sha256-counter-rejection-v1",
        "game_config": normalize_config(records[0].replay.config),
        "rules_fingerprint": rules_fingerprint(),
        "code_fingerprint": code_fingerprint(),
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValidationError(f"local complete arena report differs at {key}")
    elapsed = raw.get("elapsed_seconds")
    if (
        not isinstance(elapsed, int | float)
        or elapsed <= 0
        or raw.get("games_per_second") != pairs * 2 / elapsed
    ):
        raise ValidationError("arena runtime report differs")
    return raw


def local_nested(rows: tuple[tuple[float, ...], ...], domain: str) -> dict[str, object]:
    if len(rows) != 3 or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValidationError("local nested bootstrap rows are malformed")
    count = len(rows[0])
    seed = derive_seed_local(ROOT_SEED, f"step4:nested-bootstrap:{domain}")
    rng = LocalRandom(seed, f"step4:nested-bootstrap:{domain}")
    values: list[float] = []
    for _ in range(20_000):
        outer = tuple(rng.randbelow(3) for _ in range(3))
        values.append(
            sum(
                sum(rows[index][rng.randbelow(count)] for _ in range(count)) / count
                for index in outer
            )
            / 3
        )
    values.sort()
    return {
        "version": "m7-counterfactual-rollout-nested-bootstrap-v1",
        "unit": "training-replicate-then-paired-block",
        "resamples": 20_000,
        "outer_draws_materialized_first": True,
        "order_statistic_indices": {"lower": 499, "upper": 19_499},
        "point_estimate": sum(sum(row) / count for row in rows) / 3,
        "interval": [values[499], values[19_499]],
        "seed": seed,
        "domain": domain,
        "blocks_per_replicate": count,
    }


def local_aligned(
    left: tuple[tuple[float, ...], ...], right: tuple[tuple[float, ...], ...], domain: str
) -> dict[str, object]:
    if (
        len(left) != 3
        or len(right) != 3
        or not left[0]
        or any(len(a) != len(b) or len(a) != len(left[0]) for a, b in zip(left, right, strict=True))
    ):
        raise ValidationError("local aligned bootstrap rows are malformed")
    count = len(left[0])
    seed = derive_seed_local(ROOT_SEED, f"step4:nested-bootstrap:{domain}")
    rng = LocalRandom(seed, f"step4:nested-bootstrap:{domain}")
    values: list[float] = []
    for _ in range(20_000):
        outer = tuple(rng.randbelow(3) for _ in range(3))
        means: list[float] = []
        for replicate in outer:
            indexes = tuple(rng.randbelow(count) for _ in range(count))
            means.append(
                sum(left[replicate][index] - right[replicate][index] for index in indexes) / count
            )
        values.append(sum(means) / 3)
    values.sort()
    point = sum(sum(a) / count - sum(b) / count for a, b in zip(left, right, strict=True)) / 3
    return {
        "version": "m7-counterfactual-rollout-aligned-nested-bootstrap-v1",
        "unit": "training-replicate-then-identically-aligned-paired-block",
        "resamples": 20_000,
        "outer_draws_materialized_first": True,
        "same_inner_block_indexes_for_contrast": True,
        "order_statistic_indices": {"lower": 499, "upper": 19_499},
        "point_estimate": point,
        "interval": [values[499], values[19_499]],
        "seed": seed,
        "domain": domain,
        "blocks_per_replicate": count,
    }


def local_pair_scores(report: Mapping[str, object]) -> tuple[float, ...]:
    return tuple(
        cast(int, cast(Mapping[str, object], row)["agent_a_wins"]) / 2
        for row in cast(list[object], report["paired_seed_outcomes"])
    )


def local_claim_statistics(arenas: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    rows: dict[str, list[tuple[float, ...]]] = {
        "direct": [],
        "parent": [],
        "random": [],
        "heur_t": [],
        "heur_c": [],
        "heur_q0": [],
    }
    macro_t: list[float] = []
    macro_c: list[float] = []
    seats: list[float] = []
    tactical = True
    for replicate in ("replicate-1", "replicate-2", "replicate-3"):
        index = replicate.removeprefix("replicate-")
        rows["direct"].append(local_pair_scores(arenas[f"T{index}-vs-C{index}"]))
        rows["parent"].append(local_pair_scores(arenas[f"T{index}-vs-q0-parent"]))
        rows["random"].append(local_pair_scores(arenas[f"T{index}-vs-random"]))
        rows["heur_t"].append(local_pair_scores(arenas[f"T{index}-vs-heuristic"]))
        rows["heur_c"].append(local_pair_scores(arenas[f"C{index}-vs-heuristic"]))
        rows["heur_q0"].append(
            local_pair_scores(arenas[f"q0-parent-vs-heuristic-reference-{index}"])
        )
        opponents = ("q0-parent", "heuristic", "random", "historical-q0", "q1", "q2", "q3", "q4")
        macro_t.append(
            sum(
                cast(float, arenas[f"T{index}-vs-{opponent}"]["agent_a_win_rate"])
                for opponent in opponents
            )
            / 8
        )
        macro_c.append(
            sum(
                cast(float, arenas[f"C{index}-vs-{opponent}"]["agent_a_win_rate"])
                for opponent in opponents
            )
            / 8
        )
        for opponent in (f"C{index}", "q0-parent", "heuristic", "random"):
            by_seat = cast(
                Mapping[str, Mapping[str, object]],
                arenas[f"T{index}-vs-{opponent}"]["agent_a_by_seat"],
            )
            seats.extend(
                cast(float, by_seat[seat]["win_rate"]) for seat in ("player_one", "player_two")
            )
        tactical = tactical and all(
            cast(Mapping[str, object], value["tactical"])["passed"] is True
            for key, value in arenas.items()
            if key.startswith(f"T{index}-")
        )
    return {
        "smoke_only": False,
        "bootstrap_contract": {
            "root_seed": ROOT_SEED,
            "outer_draws_materialized_before_inner": True,
            "heuristic_contrasts_use_identical_aligned_block_indexes": True,
        },
        "nested": {
            "treatment_vs_control": local_nested(tuple(rows["direct"]), "treatment-vs-control"),
            "treatment_vs_q0_parent": local_nested(tuple(rows["parent"]), "treatment-vs-q0-parent"),
            "treatment_vs_random": local_nested(tuple(rows["random"]), "treatment-vs-random"),
            "treatment_minus_q0_parent_heuristic": local_aligned(
                tuple(rows["heur_t"]), tuple(rows["heur_q0"]), "treatment-minus-q0-parent-heuristic"
            ),
            "control_minus_q0_parent_heuristic": local_aligned(
                tuple(rows["heur_c"]), tuple(rows["heur_q0"]), "control-minus-q0-parent-heuristic"
            ),
        },
        "direct_replicate_points": [sum(row) / len(row) for row in rows["direct"]],
        "candidate_treatment_seat_floor": min(seats),
        "tactical_passed": tactical,
        "equal_macro_descriptive": {
            "treatment": sum(macro_t) / 3,
            "control": sum(macro_c) / 3,
            "per_replicate_treatment": macro_t,
            "per_replicate_control": macro_c,
        },
    }


def local_claim_selection(statistics: Mapping[str, object]) -> dict[str, object]:
    nested = cast(Mapping[str, Mapping[str, object]], statistics["nested"])
    conditions = {
        "treatment_vs_control_lower_strictly_above_50": cast(
            list[float], nested["treatment_vs_control"]["interval"]
        )[0]
        > 0.5,
        "two_of_three_direct_replicates_above_50": sum(
            value > 0.5 for value in cast(list[float], statistics["direct_replicate_points"])
        )
        >= 2,
        "treatment_vs_q0_parent_lower_above_50": cast(
            list[float], nested["treatment_vs_q0_parent"]["interval"]
        )[0]
        > 0.5,
        "treatment_vs_random_lower_above_50": cast(
            list[float], nested["treatment_vs_random"]["interval"]
        )[0]
        > 0.5,
        "heuristic_aligned_lower_strictly_above_minus_5pp": cast(
            list[float], nested["treatment_minus_q0_parent_heuristic"]["interval"]
        )[0]
        > -0.05,
        "candidate_treatment_seat_floor_at_least_45": cast(
            float, statistics["candidate_treatment_seat_floor"]
        )
        >= 0.45,
        "tactical_invariants": statistics.get("tactical_passed") is True,
        "source_split_panel_sampler_rollout_schedule_validator_integrity": True,
    }
    advances = all(conditions.values())
    direct_lower = cast(list[float], nested["treatment_vs_control"]["interval"])[0]
    return {
        "version": "m7-counterfactual-rollout-selection-v1",
        "classification": "advances_step5_recipe_family"
        if advances
        else ("does_not_advance" if direct_lower <= 0.5 else "inconclusive_does_not_advance"),
        "conditions": conditions,
        "rollout_treatment_enters_step5": advances,
        "step5_disposition": (
            "all three rollout treatments enter Step-5 as one family; "
            "controls remain paired references"
            if advances
            else (
                "rollout treatment does not advance; retained Step-3 M-v2 remains "
                "the descriptive Step-5 entry"
            )
        ),
        "step5_entries": ["M-v2-rollout-treatment-family:T1,T2,T3"]
        if advances
        else ["retained-Step3-M-v2-descriptive"],
        "equal_macro_descriptive": statistics.get("equal_macro_descriptive"),
    }


def validate_arenas_and_stats(output: Path, plan: Mapping[str, object], smoke: bool) -> None:
    result = read(output / "result.json")
    retained = cast(Mapping[str, object], result["arenas"])
    declared = cast(list[Mapping[str, object]], cast(Mapping[str, object], plan["arenas"])["cells"])
    if set(retained) != {cast(str, cell["key"]) for cell in declared}:
        raise ValidationError("result arena keys do not cover the declared schedule")
    if (smoke and len(declared) != 1) or (
        not smoke
        and (
            len(declared) != 54
            or cast(Mapping[str, object], plan["arenas"])["total_physical_games"] != 24_000
        )
    ):
        raise ValidationError("arena cell/game cardinality differs")
    shared_setups: dict[tuple[str, str], tuple[int, ...]] = {}
    local_reports: dict[str, Mapping[str, object]] = {}
    for cell in declared:
        key = cast(str, cell["key"])
        report = read(output / "arenas" / key / "report.json")
        artifact(report, f"arena {key}")
        raw = local_validate_arena(output / "arenas" / key, report, cell)
        if cell.get("candidate") == "T":
            _, treatment_records = load_corpus(
                output / "arenas" / key / "records", verify_code=False, verify_replays=True
            )
            local_tactical = local_treatment_tactical(treatment_records)
            retained_tactical = cast(Mapping[str, object], report["tactical"])
            checked = cast(Mapping[str, Mapping[str, object]], retained_tactical["checked_agents"])
            for agent_id, counts in local_tactical.items():
                retained_counts = checked[agent_id]
                if (
                    retained_counts.get("avoidable_losses") != counts["avoidable_losses"]
                    or retained_counts.get("missed_guaranteed_wins")
                    != counts["missed_guaranteed_wins"]
                    or retained_counts.get("passed")
                    != (counts["avoidable_losses"] == 0 and counts["missed_guaranteed_wins"] == 0)
                ):
                    raise ValidationError("independent treatment tactical audit differs")
            if any(
                value["avoidable_losses"] or value["missed_guaranteed_wins"]
                for value in local_tactical.values()
            ):
                raise ValidationError("treatment tactical invariant is violated")
        local_reports[key] = {**raw, "tactical": report["tactical"]}
        shared = cell.get("shared_group")
        if isinstance(shared, str):
            master = local_expected_master_seed(cell)
            setups = tuple(
                derive_seed_local(master, f"arena:pair:{index}:setup") & ((1 << 64) - 1)
                for index in range(cast(int, cell["paired_blocks"]))
            )
            group_key = (cast(str, cell["replicate_id"]), shared)
            prior = shared_setups.get(group_key)
            if prior is not None and prior != setups:
                raise ValidationError("shared-group arena setups are not aligned")
            shared_setups[group_key] = setups
    statistics = read(output / "statistics.json")
    selection = read(output / "selection.json")
    artifact(statistics, "statistics")
    artifact(selection, "selection")
    if smoke:
        if (
            selection.get("classification") != "nonclaim_smoke_no_disposition"
            or selection.get("rollout_treatment_enters_step5") is not False
        ):
            raise ValidationError("smoke selection disposition differs")
        return
    expected_statistics = local_claim_statistics(local_reports)
    expected_statistics["version"] = "m7-counterfactual-rollout-statistics-v1"
    actual_statistics = {
        key: value for key, value in statistics.items() if key != "artifact_fingerprint"
    }
    if actual_statistics != expected_statistics:
        raise ValidationError("local claim nested statistics differ")
    expected_selection = local_claim_selection(expected_statistics)
    actual_selection = {
        key: value for key, value in selection.items() if key != "artifact_fingerprint"
    }
    if actual_selection != expected_selection:
        raise ValidationError("local claim gate/disposition differs")


CHECKSUM_EXCLUDED_NAMES = frozenset(
    {
        "result.json",
        "execution-state.json",
        "checksums.json",
        "validation.json",
        "driver.stdout",
        "driver.stderr",
        "wrapper-exit.json",
        "validation.stdout",
        "validation.stderr",
        "validator-runtime.json",
        "preflight-runtime.json",
    }
)


def local_checksum_scope(output: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in CHECKSUM_EXCLUDED_NAMES or path.name == ".lock":
            continue
        entries[path.relative_to(output).as_posix()] = sha256(path)
    return entries


def validate_result_cross_references(output: Path, plan: Mapping[str, object]) -> None:
    result = read(output / "result.json")
    if (
        result.get("version") != "m7-counterfactual-rollout-result-v1"
        or result.get("cycle_id") != CYCLE_ID
        or result.get("status") != "completed"
        or result.get("plan_fingerprint") != plan.get("plan_fingerprint")
        or result.get("input_manifest_fingerprint") != plan.get("input_manifest_fingerprint")
        or result.get("result_fingerprint")
        != digest({key: value for key, value in result.items() if key != "result_fingerprint"})
    ):
        raise ValidationError("result identity/version/status/fingerprint differs")
    replicates = ("replicate-1", "replicate-2", "replicate-3")
    panels = cast(Mapping[str, object], result.get("panels"))
    rollouts = cast(Mapping[str, object], result.get("rollouts"))
    if set(panels) != set(replicates) or set(rollouts) != set(replicates):
        raise ValidationError("result panel/rollout replicate cardinality differs")
    training = read(output / "training" / "summary.json")
    artifact(training, "training summary")
    if result.get("training_fingerprint") != training.get("artifact_fingerprint"):
        raise ValidationError("result training cross-reference differs")
    rows = training.get("replicates")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValidationError("training summary lacks three paired replicates")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValidationError("training summary row is malformed")
        replicate = row.get("replicate_id")
        if replicate not in replicates or replicate in seen:
            raise ValidationError("training replicate cardinality differs")
        seen.add(cast(str, replicate))
        for arm in ("control", "treatment"):
            entry = row.get(arm)
            if not isinstance(entry, Mapping):
                raise ValidationError("training arm entry is missing")
            history = read(output / "training" / arm / cast(str, replicate) / "history.json")
            artifact(history, f"{arm} training history")
            if history.get("artifact_fingerprint") != entry.get("artifact_fingerprint"):
                raise ValidationError("training history cross-reference differs")
            checkpoint = inspect_structured_checkpoint(
                output / "checkpoints" / arm / cast(str, replicate)
            )
            if checkpoint.checkpoint_fingerprint != entry.get(
                "checkpoint_fingerprint"
            ) or checkpoint.tensor_digest != entry.get("tensor_digest"):
                raise ValidationError("checkpoint/training tensor cross-reference differs")
        reproduction = read(output / "control-reproduction" / cast(str, replicate) / "summary.json")
        artifact(reproduction, "control reproduction")
        if reproduction.get("artifact_fingerprint") != cast(
            Mapping[str, object], row["reproduction"]
        ).get("artifact_fingerprint"):
            raise ValidationError("control reproduction cross-reference differs")
        if reproduction.get("byte_for_byte_tensor_match") is not True:
            raise ValidationError("control reproduction is not exact")
    for replicate in replicates:
        panel = read(output / "panel" / replicate / "manifest.json")
        rollout = read(output / "rollouts" / replicate / "manifest.json")
        artifact(panel, "panel manifest")
        artifact(rollout, "rollout manifest")
        if panels.get(replicate) != panel.get("artifact_fingerprint") or rollouts.get(
            replicate
        ) != rollout.get("artifact_fingerprint"):
            raise ValidationError("result panel/rollout manifest cross-reference differs")
    cells = cast(list[Mapping[str, object]], cast(Mapping[str, object], plan["arenas"])["cells"])
    arena_result = cast(Mapping[str, object], result.get("arenas"))
    if set(arena_result) != {cast(str, cell["key"]) for cell in cells}:
        raise ValidationError("result arena cardinality/cross-reference differs")
    for cell in cells:
        key = cast(str, cell["key"])
        report = read(output / "arenas" / key / "report.json")
        artifact(report, "arena report")
        if arena_result.get(key) != report.get("artifact_fingerprint"):
            raise ValidationError("result arena report reference differs")
    for filename, key in (
        ("statistics.json", "statistics_fingerprint"),
        ("safety-report.json", "safety_fingerprint"),
        ("selection.json", "selection_fingerprint"),
        ("runtime-extrapolation.json", "runtime_fingerprint"),
        ("checksums.json", "checksums_fingerprint"),
    ):
        value = read(output / filename)
        artifact(value, filename)
        if result.get(key) != value.get("artifact_fingerprint"):
            raise ValidationError(f"result {filename} cross-reference differs")


def local_treatment_tactical(records: Iterable[GameRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        state = new_game(record.replay.config, record.replay.seed)
        for action in record.replay.actions:
            actor = (
                state.active_player if state.phase is Phase.PLAY else state.active_player.other()
            )
            seat = record.seats[0 if actor is PlayerId.PLAYER_ONE else 1]
            if seat.agent_id.startswith("T-replicate-"):
                observation = local_engine_observation(state, actor)
                safety = filter_terminal_actions(observation, observation.legal_actions)
                offense = filter_immediate_win_actions(observation, observation.legal_actions)
                row = counts.setdefault(
                    seat.agent_id,
                    {"avoidable_losses": 0, "missed_guaranteed_wins": 0},
                )
                row["avoidable_losses"] += int(
                    action in safety.provable_loss_actions and not safety.forced_loss_fallback
                )
                row["missed_guaranteed_wins"] += int(
                    bool(offense.forced_win_actions) and action not in offense.forced_win_actions
                )
            state = apply_action(state, action)
    return counts


def validate_checksums(output: Path) -> None:
    checksums = read(output / "checksums.json")
    artifact(checksums, "checksums")
    files = checksums.get("files")
    if not isinstance(files, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in files.items()
    ):
        raise ValidationError("checksum file map is malformed")
    expected = local_checksum_scope(output)
    if dict(files) != expected:
        raise ValidationError("checksum scope has missing, extra, or mismatched payload paths")


def _number(value: object, label: str) -> float:
    if not isinstance(value, int | float):
        raise ValidationError(f"{label} must be numeric")
    return float(value)


def write_runtime_preflight(
    output: Path,
    plan: Mapping[str, object],
    timings: Mapping[str, float],
    target: Mapping[str, object],
) -> dict[str, object]:
    production = read(output / "runtime-extrapolation.json")
    artifact(production, "production runtime")
    production_minutes = _number(production.get("projected_full_minutes"), "production minutes")
    production_units = _number(production.get("rollout_units_per_second"), "production units/s")
    production_leaf = _number(production.get("leaf_fraction"), "production leaf fraction")
    target_units = _number(target.get("target_work_units"), "validator target units")
    target_seconds = _number(target.get("target_elapsed_seconds"), "validator target seconds")
    target_rate = _number(target.get("target_units_per_second"), "validator target units/s")
    execution = cast(Mapping[str, object], plan["execution"])
    claim_mode = execution.get("claim") is True
    full_target_units = 42_000 if not claim_mode else target_units
    target_ratio = full_target_units / target_units
    current_games = _number(
        cast(Mapping[str, object], plan["arenas"])["total_physical_games"], "current arena games"
    )
    arena_ratio = 24_000 / current_games
    statistics_seconds = timings.get("statistics", 0.0) if claim_mode else 30 * 60
    independent_full_seconds = (
        timings.get("checksums", 0.0)
        + timings.get("panel", 0.0)
        + timings.get("controls", 0.0)
        + target_seconds * target_ratio
        + timings.get("arenas_stats", 0.0) * arena_ratio
        + statistics_seconds
    )
    combined_minutes = production_minutes + independent_full_seconds / 60
    target_leaf = _number(target.get("depth_leaf_fraction"), "validator leaf fraction")
    target_caps = target.get("mechanical_cap_errors")
    if type(target_caps) is not int:
        raise ValidationError("validator mechanical cap count is malformed")
    result = {
        "version": "m7-counterfactual-rollout-independent-runtime-preflight-v1",
        "source_code_fingerprint": cast(Mapping[str, object], plan["source"])["code_fingerprint"],
        "input_manifest_fingerprint": plan["input_manifest_fingerprint"],
        "validator_phase_seconds": dict(timings),
        "validator_target_work_units": target_units,
        "validator_target_units_per_second": target_rate,
        "validator_target_leaf_fraction": target_leaf,
        "validator_mechanical_cap_errors": target_caps,
        "validator_statistics_full_projection_seconds": statistics_seconds,
        "validator_arena_projection_ratio": arena_ratio,
        "validator_target_projection_ratio": target_ratio,
        "validator_full_projection_minutes": independent_full_seconds / 60,
        "production_full_projection_minutes": production_minutes,
        "combined_full_projection_minutes": combined_minutes,
        "claim_cutoff_minutes": 465,
        "production_units_per_second": production_units,
        "both_target_throughputs_at_least_67": production_units >= 67 and target_rate >= 67,
        "leaf_and_cap_gate": production_leaf <= 0.5 and target_leaf <= 0.5 and target_caps == 0,
        "claim_eligible": production_units >= 67
        and target_rate >= 67
        and production_leaf <= 0.5
        and target_leaf <= 0.5
        and target_caps == 0
        and combined_minutes < 465,
    }
    result["artifact_fingerprint"] = digest(result)
    (output / "preflight-runtime.json").write_bytes(canonical(result) + b"\n")
    return result


def validate(output: Path, step3_root: Path) -> dict[str, object]:
    started = time.perf_counter()
    static_source_guard()
    plan = read(output / "plan.json")
    if plan.get("cycle_id") != CYCLE_ID:
        raise ValidationError("wrong cycle")
    smoke = cast(Mapping[str, object], plan["execution"]).get("claim") is False
    timings: dict[str, float] = {}
    stage = time.perf_counter()
    validate_checksums(output)
    validate_result_cross_references(output, plan)
    timings["checksums"] = time.perf_counter() - stage
    stage = time.perf_counter()
    validate_panels(output, step3_root, plan)
    timings["panel"] = time.perf_counter() - stage
    target = validate_targets(output, plan)
    timings["targets"] = _number(target["target_elapsed_seconds"], "target elapsed")
    stage = time.perf_counter()
    validate_controls(output, step3_root, plan)
    timings["controls"] = time.perf_counter() - stage
    stage = time.perf_counter()
    validate_arenas_and_stats(output, plan, smoke)
    timings["arenas_stats"] = time.perf_counter() - stage
    timings["total"] = time.perf_counter() - started
    runtime = write_runtime_preflight(output, plan, timings, target)
    return {
        "version": "m7-counterfactual-rollout-independent-validation-v2",
        "plan_fingerprint": plan["plan_fingerprint"],
        "status": "passed",
        "runtime_preflight_fingerprint": runtime["artifact_fingerprint"],
        "checks": {
            "source_guard": True,
            "input_control_reproduction": True,
            "local_panel_quotas_and_identities": True,
            "local_latent_sampler_transition_target_recomputation": True,
            "local_transcript_recomputation_and_quarantine": True,
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
