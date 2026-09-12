#!/usr/bin/env python3
"""Independent local validator for M7 Step-3 structured-model-v2 evidence.

This file intentionally does not import the production structured encoder, Step-3 runner,
production arena aggregation, production statistics, or production selection.  Its structured
features are rebuilt from public observations plus the pure terminal evaluator.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np

from agent_avenue.agents import (
    RNG_ALGORITHM,
    SEED_DERIVATION,
    DeterministicRandom,
    derive_seed,
    filter_immediate_win_actions,
    filter_terminal_actions,
)
from agent_avenue.encoding.schema import CARD_ORDER
from agent_avenue.engine import Phase, PlayerId, apply_action, new_game
from agent_avenue.engine.cards import CARD_DEFINITIONS, CardEffect, CardName, recruit_effect
from agent_avenue.engine.model import (
    Action,
    OfferSlot,
    PlayOfferAction,
    RecruitAction,
    TerminalOutcome,
)
from agent_avenue.engine.setup import normalize_config
from agent_avenue.engine.terminal import adjudicate_position
from agent_avenue.learning import (
    load_checkpoint,
    load_structured_checkpoint,
    load_structured_dataset,
    tensor_digest,
)
from agent_avenue.observation import observe
from agent_avenue.observation.model import PlayerObservation, PublicPlayer, RecruitContext
from agent_avenue.storage import (
    code_fingerprint,
    game_record_fingerprint,
    inspect_source_identity,
    load_corpus,
    repository_root,
    rules_fingerprint,
    verify_game_record,
)

CYCLE_ID = "m7-structured-model-v2"
ROOT_SEED = 2026091203
REPLICATES = ("replicate-1", "replicate-2", "replicate-3")
ARMS = ("C", "M")
OPPONENTS = ("q0-parent", "heuristic", "random", "historical-q0", "q1", "q2", "q3", "q4")
RESAMPLES = 20_000
LOWER = 499
UPPER = 19_499


V1_PREFIX_WIDTH = 87
_V1_REPEATABLE_SCORE_CARDS = (
    CardName.DOUBLE_AGENT,
    CardName.ENFORCER,
    CardName.SABOTEUR,
    CardName.SENTINEL,
)
_V1_THRESHOLDS = (
    *((card, (1, 2, 3)) for card in _V1_REPEATABLE_SCORE_CARDS),
    (CardName.CODEBREAKER, (1, 2)),
    (CardName.DAREDEVIL, (1, 2)),
    (CardName.SIDEKICK, (1,)),
    (CardName.MOLE, (1,)),
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
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValidationError(f"artifact is not an object: {path}")
    return cast(dict[str, object], value)


def check_artifact(value: Mapping[str, object], name: str) -> None:
    actual = value.get("artifact_fingerprint")
    expected = digest({key: item for key, item in value.items() if key != "artifact_fingerprint"})
    if actual != expected:
        raise ValidationError(f"{name} fingerprint mismatch")


def number(value: object, label: str) -> float:
    if not isinstance(value, int | float):
        raise ValidationError(f"{label} must be numeric")
    return float(value)


# ---- Validator-local structured feature calculation: 519 frozen values. ----


def f32(value: float) -> float:
    result = struct.unpack("!f", struct.pack("!f", value))[0]
    if not math.isfinite(result):
        raise ValidationError("non-finite local structured feature")
    return float(result)


def one_hot(card: CardName | None) -> list[float]:
    values = [0.0] * len(CARD_ORDER)
    if card is not None:
        values[CARD_ORDER.index(card)] = 1.0
    return values


def public_player(observation: PlayerObservation, player: PlayerId) -> PublicPlayer:
    rows = tuple(value for value in observation.players if value.player is player)
    if len(rows) != 1:
        raise ValidationError("safe observation lacks a unique public player")
    return rows[0]


def effect_kind(effect: CardEffect | None) -> list[float]:
    values = [0.0, 0.0, 0.0]
    if effect is not None:
        values[("score", "win", "lose").index(effect.kind)] = 1.0
    return values


def score(effect: CardEffect) -> int:
    return effect.points if effect.kind == "score" else 0


def effect_for(player: PublicPlayer, card: CardName) -> tuple[CardEffect, int, int]:
    prior = player.recruited.count(card)
    return recruit_effect(card, prior + 1), prior, prior + 1


def terminal(
    observation: PlayerObservation,
    self_card: CardName,
    other_card: CardName,
    *,
    deck_empty: bool,
    next_hand: int,
) -> tuple[TerminalOutcome | None, CardEffect, CardEffect]:
    self_player = public_player(observation, observation.viewer)
    other_player = public_player(observation, observation.viewer.other())
    self_effect, _, _ = effect_for(self_player, self_card)
    other_effect, _, _ = effect_for(other_player, other_card)
    if observation.viewer is PlayerId.PLAYER_ONE:
        recruited = ((*self_player.recruited, self_card), (*other_player.recruited, other_card))
        scores = (self_player.score + score(self_effect), other_player.score + score(other_effect))
    else:
        recruited = ((*other_player.recruited, other_card), (*self_player.recruited, self_card))
        scores = (other_player.score + score(other_effect), self_player.score + score(self_effect))
    return (
        adjudicate_position(
            scores=scores,
            recruited=recruited,
            active_player=observation.active_player,
            turn=observation.turn,
            deck_empty=deck_empty,
            next_player_hand_size=next_hand,
        ),
        self_effect,
        other_effect,
    )


def terminal_bits(
    outcome: TerminalOutcome | None, viewer: PlayerId
) -> tuple[list[float], list[float]]:
    if outcome is None:
        return [0.0, 0.0], [0.0, 0.0]
    return ([1.0, 0.0], [0.0, 1.0]) if outcome.winner is viewer else ([0.0, 1.0], [1.0, 0.0])


def history_features(observation: PlayerObservation) -> list[float]:
    history = observation.history
    if type(history) is not tuple or len(history) > 19:
        raise ValidationError("public history shape is malformed")
    values: list[float] = [0.0] * ((8 - len(history[-8:])) * 39)
    for event in history[-8:]:
        if event.opponent_recruited is not (
            event.face_up if event.chosen_slot is OfferSlot.FACE_UP else event.face_down
        ):
            raise ValidationError("completed public turn recruit assignment is malformed")
        self_change = event.score_changes[0 if observation.viewer is PlayerId.PLAYER_ONE else 1]
        other_change = event.score_changes[1 if observation.viewer is PlayerId.PLAYER_ONE else 0]
        values.extend(
            (
                1.0,
                float(event.active_player is observation.viewer),
                float(event.active_player is observation.viewer.other()),
            )
        )
        values.extend(one_hot(event.face_up))
        values.extend(one_hot(event.face_down))
        values.extend(
            (
                float(event.chosen_slot is OfferSlot.FACE_UP),
                float(event.chosen_slot is OfferSlot.FACE_DOWN),
            )
        )
        values.extend(one_hot(event.opponent_recruited))
        values.extend(one_hot(event.active_recruited))
        values.extend(
            (f32(max(-3, min(6, self_change)) / 6), f32(max(-3, min(6, other_change)) / 6))
        )
    values.extend((f32(len(history) / 19), f32(max(len(history) - 8, 0) / 19)))
    if len(values) != 314:
        raise ValidationError("local history width mismatch")
    return values


def play_branch(
    observation: PlayerObservation,
    self_card: CardName,
    other_card: CardName,
    *,
    deck_empty: bool,
    next_hand: int,
) -> list[float]:
    self_player = public_player(observation, observation.viewer)
    other_player = public_player(observation, observation.viewer.other())
    outcome, self_effect, other_effect = terminal(
        observation, self_card, other_card, deck_empty=deck_empty, next_hand=next_hand
    )
    self_bits, other_bits = terminal_bits(outcome, observation.viewer)
    _, _, self_count = effect_for(self_player, self_card)
    _, _, other_count = effect_for(other_player, other_card)
    values = (
        one_hot(self_card)
        + one_hot(other_card)
        + effect_kind(self_effect)
        + effect_kind(other_effect)
        + self_bits
        + other_bits
    )
    values.extend(
        (
            f32(score(self_effect) / 6),
            f32(score(other_effect) / 6),
            f32(self_count / CARD_DEFINITIONS[self_card].copies),
            f32(other_count / CARD_DEFINITIONS[other_card].copies),
        )
    )
    return values


def play_features(observation: PlayerObservation, action: PlayOfferAction) -> list[float]:
    self_player = public_player(observation, observation.viewer)
    other_player = public_player(observation, observation.viewer.other())
    after_offer = self_player.hand_size - 2
    refill = min(4 - after_offer, observation.remaining_deck_count)
    deck_empty = observation.remaining_deck_count - refill == 0
    values = play_branch(
        observation,
        action.face_down,
        action.face_up,
        deck_empty=deck_empty,
        next_hand=other_player.hand_size,
    )
    values.extend(
        play_branch(
            observation,
            action.face_up,
            action.face_down,
            deck_empty=deck_empty,
            next_hand=other_player.hand_size,
        )
    )
    values.extend(
        (
            f32(after_offer / 4),
            f32(refill / 4),
            f32((after_offer + refill) / 4),
            float(action.face_up is action.face_down),
        )
    )
    if len(values) != 64:
        raise ValidationError("local play width mismatch")
    return values


def safe_unseen(observation: PlayerObservation) -> tuple[int, ...]:
    visible = Counter(observation.own_hand)
    for player in observation.players:
        visible.update(player.recruited)
    if isinstance(observation.decision, RecruitContext):
        visible[observation.decision.face_up] += 1
    return tuple(CARD_DEFINITIONS[card].copies - visible[card] for card in CARD_ORDER)


def local_v1_prefix(observation: PlayerObservation, action: Action) -> tuple[float, ...]:
    """Independently rebuild frozen candidate-public-v1's 87 safe features.

    Records have already passed semantic replay verification; these local checks intentionally
    validate the public accounting needed by the prefix without importing any production encoder.
    """
    if action not in observation.legal_actions:
        raise ValidationError("candidate is not a legal safe-observation action")
    self_player = public_player(observation, observation.viewer)
    other_player = public_player(observation, observation.viewer.other())
    if self_player.hand_size != len(observation.own_hand):
        raise ValidationError("local v1 self hand accounting differs")
    if not isinstance(observation.decision, RecruitContext) and observation.phase is not Phase.PLAY:
        raise ValidationError("local v1 play phase differs")
    if isinstance(observation.decision, RecruitContext) and observation.phase is not Phase.RECRUIT:
        raise ValidationError("local v1 recruit phase differs")
    if (
        observation.turn < 1
        or observation.turn > 19
        or not 0 <= observation.remaining_deck_count <= 30
    ):
        raise ValidationError("local v1 turn/deck bounds differ")
    visible = Counter(observation.own_hand)
    visible.update(self_player.recruited)
    visible.update(other_player.recruited)
    if isinstance(observation.decision, RecruitContext):
        visible[observation.decision.face_up] += 1
    unseen = tuple(CARD_DEFINITIONS[card].copies - visible[card] for card in CARD_ORDER)
    expected_unknown = observation.remaining_deck_count + other_player.hand_size
    if isinstance(observation.decision, RecruitContext):
        expected_unknown += 1
    if any(count < 0 for count in unseen) or sum(unseen) != expected_unknown:
        raise ValidationError("local v1 safe unseen accounting differs")

    def clip(value: int, limit: int) -> float:
        return float(max(-limit, min(limit, value)) / limit)

    values: list[float] = [
        float(observation.phase is Phase.PLAY),
        float(observation.phase is Phase.RECRUIT),
        float(observation.turn / 19),
        float(observation.remaining_deck_count / 30),
        clip(self_player.score, 14),
        clip(other_player.score, 14),
        clip(self_player.score - other_player.score, 7),
        float(self_player.hand_size / 4),
        float(other_player.hand_size / 4),
    ]
    hand = Counter(observation.own_hand)
    values.extend(float(hand[card] / CARD_DEFINITIONS[card].copies) for card in CARD_ORDER)
    for recruited in (self_player.recruited, other_player.recruited):
        counts = Counter(recruited)
        values.extend(
            float(counts[card] >= threshold)
            for card, thresholds in _V1_THRESHOLDS
            for threshold in thresholds
        )
    values.extend(
        float(count / CARD_DEFINITIONS[card].copies)
        for card, count in zip(CARD_ORDER, unseen, strict=True)
    )
    values.extend(
        one_hot(
            observation.decision.face_up
            if isinstance(observation.decision, RecruitContext)
            else None
        )
    )
    values.extend(one_hot(action.face_up if isinstance(action, PlayOfferAction) else None))
    values.extend(one_hot(action.face_down if isinstance(action, PlayOfferAction) else None))
    values.extend(
        (
            float(isinstance(action, RecruitAction) and action.slot is OfferSlot.FACE_UP),
            float(isinstance(action, RecruitAction) and action.slot is OfferSlot.FACE_DOWN),
        )
    )
    if len(values) != V1_PREFIX_WIDTH:
        raise ValidationError("local v1 prefix width differs")
    return tuple(values)


def recruit_features(observation: PlayerObservation, action: RecruitAction) -> list[float]:
    if not isinstance(observation.decision, RecruitContext):
        raise ValidationError("recruit action has no public recruit context")
    known = observation.decision.face_up
    self_player = public_player(observation, observation.viewer)
    other_player = public_player(observation, observation.viewer.other())
    self_known = action.slot is OfferSlot.FACE_UP
    known_player = self_player if self_known else other_player
    known_effect, known_prior, known_count = effect_for(known_player, known)
    support = tuple(
        (card, count)
        for card, count in zip(CARD_ORDER, safe_unseen(observation), strict=True)
        if count > 0
    )
    if not support:
        raise ValidationError("empty public-consistent hidden support")
    total = sum(count for _, count in support)
    values = one_hot(known if self_known else None) + one_hot(known if not self_known else None)
    values.extend((float(not self_known), float(self_known)))
    values.extend(effect_kind(known_effect if self_known else None))
    values.extend(effect_kind(known_effect if not self_known else None))
    outcomes: list[tuple[int, TerminalOutcome | None, CardEffect]] = []
    for hidden, multiplicity in support:
        own = known if self_known else hidden
        other = hidden if self_known else known
        outcome, _, _ = terminal(
            observation,
            own,
            other,
            deck_empty=observation.remaining_deck_count == 0,
            next_hand=self_player.hand_size,
        )
        hidden_player = other_player if self_known else self_player
        hidden_effect, _, _ = effect_for(hidden_player, hidden)
        outcomes.append((multiplicity, outcome, hidden_effect))
    self_results = [terminal_bits(outcome, observation.viewer)[0] for _, outcome, _ in outcomes]
    other_results = [terminal_bits(outcome, observation.viewer)[1] for _, outcome, _ in outcomes]
    values.extend(
        (
            float(all(bits == [1.0, 0.0] for bits in self_results)),
            float(all(bits == [0.0, 1.0] for bits in self_results)),
            float(all(bits == [1.0, 0.0] for bits in other_results)),
            float(all(bits == [0.0, 1.0] for bits in other_results)),
        )
    )
    values.extend(
        (
            f32(score(known_effect) / 6) if self_known else 0.0,
            f32(score(known_effect) / 6) if not self_known else 0.0,
            f32(known_count / CARD_DEFINITIONS[known].copies) if self_known else 0.0,
            f32(known_count / CARD_DEFINITIONS[known].copies) if not self_known else 0.0,
            f32(known_prior / CARD_DEFINITIONS[known].copies),
        )
    )
    mass = {kind: 0 for kind in ("score", "win", "lose")}
    bits = {kind: 0.0 for kind in ("score", "win", "lose")}
    scores: list[int] = []
    for multiplicity, _, effect in outcomes:
        mass[effect.kind] += multiplicity
        bits[effect.kind] = 1.0
        scores.append(score(effect))
    fractions = [f32(mass[kind] / total) for kind in ("score", "win", "lose")]
    minimum, maximum = f32(min(scores) / 6), f32(max(scores) / 6)
    if self_known:
        values.extend(
            [0.0] * 3
            + fractions
            + [0.0, 0.0, minimum, maximum]
            + [0.0, f32(1 / total)]
            + [0.0] * 3
            + [bits[kind] for kind in ("score", "win", "lose")]
        )
    else:
        values.extend(
            fractions
            + [0.0] * 3
            + [minimum, maximum, 0.0, 0.0]
            + [f32(1 / total), 0.0]
            + [bits[kind] for kind in ("score", "win", "lose")]
            + [0.0] * 3
        )
    values.extend(
        (
            f32(total / 38),
            float(any(bits == [1.0, 0.0] for bits in self_results)),
            float(any(bits == [0.0, 1.0] for bits in self_results)),
        )
    )
    if len(values) != 54:
        raise ValidationError("local recruit width mismatch")
    return values


def local_encode(observation: PlayerObservation, action: Action) -> tuple[float, ...]:
    prefix = local_v1_prefix(observation, action)
    history = history_features(observation)
    if isinstance(action, PlayOfferAction):
        values = prefix + tuple(history) + tuple(play_features(observation, action)) + (0.0,) * 54
    elif isinstance(action, RecruitAction):
        values = (
            prefix + tuple(history) + (0.0,) * 64 + tuple(recruit_features(observation, action))
        )
    else:
        raise ValidationError("unsupported action")
    if len(values) != 519:
        raise ValidationError("local structured width mismatch")
    return values


def source_guard() -> None:
    source = Path(__file__).read_text()
    tree = ast.parse(source)
    forbidden_modules = {
        "agent_avenue.encoding.candidate_structured_v2",
        "agent_avenue.runners.structured_experiment",
        "agent_avenue.runners.arena",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in forbidden_modules:
            raise ValidationError(f"validator imports forbidden production module {node.module}")
        if isinstance(node, ast.ImportFrom) and any(
            alias.name
            in {"structured_statistics", "structured_selection", "arena_report_from_records"}
            for alias in node.names
        ):
            raise ValidationError("validator imports forbidden production entry point")


def replay_dataset(corpus: Path, v1_dataset: Path, v2_dataset: Path) -> None:
    manifest, records = load_corpus(corpus, verify_code=False)
    legacy = json.loads(v1_dataset.with_suffix(".json").read_text())
    source_order = cast(list[object], legacy["source_record_fingerprints"])
    split = cast(Mapping[str, object], legacy["split"])
    train_ids = set(cast(list[str], split["train_game_fingerprints"]))
    by_fingerprint = {game_record_fingerprint(record): record for record in records}
    v2 = load_structured_dataset(v2_dataset)
    rebuilt: dict[str, list[tuple[tuple[float, ...], float, int, str, str, int]]] = {
        "train": [],
        "validation": [],
    }
    for game_index, fingerprint in enumerate(source_order):
        if not isinstance(fingerprint, str):
            raise ValidationError("v1 source ordering is malformed")
        record = by_fingerprint[fingerprint]
        final_state = verify_game_record(record, verify_code=False)
        if final_state.outcome is None:
            raise ValidationError("nonterminal historical replay")
        winner = final_state.outcome.winner
        state = new_game(record.replay.config, record.replay.seed)
        destination = "train" if fingerprint in train_ids else "validation"
        for decision_index, action in enumerate(record.replay.actions):
            actor = (
                state.active_player if state.phase is Phase.PLAY else state.active_player.other()
            )
            vector = local_encode(observe(state, actor), action)
            rebuilt[destination].append(
                (
                    vector,
                    float(actor is winner),
                    game_index,
                    state.phase.value,
                    fingerprint,
                    decision_index,
                )
            )
            state = apply_action(state, action)
    for name, data in rebuilt.items():
        current = v2.train if name == "train" else v2.validation
        features = np.asarray([row[0] for row in data], dtype=np.float32)
        if not (
            np.array_equal(features, current.features)
            and np.array_equal(
                np.asarray([row[1] for row in data], dtype=np.float32), current.targets
            )
            and np.array_equal(
                np.asarray([row[2] for row in data], dtype=np.int64), current.game_index
            )
            and np.array_equal(np.asarray([row[3] for row in data], dtype=np.str_), current.phase)
            and np.array_equal(
                np.asarray([row[4] for row in data], dtype=np.str_), current.record_fingerprint
            )
            and np.array_equal(
                np.asarray([row[5] for row in data], dtype=np.int64), current.decision_index
            )
        ):
            raise ValidationError(f"local v2 feature/row reconstruction differs for {name}")
    if manifest.corpus_fingerprint != v2.manifest.get("source_corpus_fingerprint"):
        raise ValidationError("v2 corpus lineage mismatch")


def local_tactical(records: Sequence[object]) -> None:
    for raw in records:
        record = cast(object, raw)
        replay = record.replay
        seats = record.seats
        state = new_game(replay.config, replay.seed)
        for action in replay.actions:
            actor = (
                state.active_player if state.phase is Phase.PLAY else state.active_player.other()
            )
            seat = seats[0 if actor is PlayerId.PLAYER_ONE else 1]
            observation = observe(state, actor)
            # Canonical random/heuristic/historical comparators are intentionally unwrapped.
            if seat.config.get("type") == "terminal_offense":
                safety = filter_terminal_actions(observation, observation.legal_actions)
                offense = filter_immediate_win_actions(observation, observation.legal_actions)
                if action in safety.provable_loss_actions and not safety.forced_loss_fallback:
                    raise ValidationError("learned envelope executed avoidable immediate loss")
                if offense.forced_win_actions and action not in offense.forced_win_actions:
                    raise ValidationError("learned envelope missed guaranteed current-turn win")
            state = apply_action(state, action)


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
    return max(0.0, center - radius), min(1.0, center + radius)


def local_pair_bootstrap(pair_wins: tuple[int, ...], master_seed: int) -> dict[str, object]:
    domain = "arena:paired-bootstrap:agent-a-win-rate:v1"
    seed = derive_seed(master_seed, domain)
    rng = DeterministicRandom(seed, domain)
    totals = [
        sum(pair_wins[rng.randbelow(len(pair_wins))] for _ in pair_wins) for _ in range(20_000)
    ]
    totals.sort()
    return {
        "method": "paired-percentile-bootstrap-v1",
        "statistic": "agent_a_win_rate",
        "unit": "two-game paired seed block",
        "confidence_level": 0.95,
        "point_estimate": sum(pair_wins) / (2 * len(pair_wins)),
        "pair_count": len(pair_wins),
        "resample_count": 20_000,
        "order_statistic_indices": {"lower": 499, "upper": 19_499},
        "interval": [totals[499] / (2 * len(pair_wins)), totals[19_499] / (2 * len(pair_wins))],
        "bootstrap_seed": seed,
        "bootstrap_rng_domain": domain,
        "rng_algorithm": RNG_ALGORITHM,
        "seed_derivation": SEED_DERIVATION,
    }


def expected_agent_id(cell: Mapping[str, object]) -> tuple[str, str, str, str]:
    key = cast(str, cell["key"])
    replicate = cast(str, cell["replicate_id"])
    index = replicate.removeprefix("replicate-")
    if key.endswith("v2-vs-v1"):
        arm = cast(str, cell["candidate_arm"])
        item = f"{arm}{index}"
        return (
            f"{item}-v2",
            f"{item}-v1",
            f"step3:{replicate}:architecture-v2",
            f"step3:{replicate}:architecture-v1",
        )
    if "-v2-vs-C" in key:
        return (
            f"M{index}-v2",
            f"C{index}-v2",
            f"step3:{replicate}:M-candidate",
            f"step3:{replicate}:C-candidate",
        )
    if cast(str, cell["candidate_kind"]) == "parent":
        return (
            "q0-parent",
            "heuristic",
            f"step3:{replicate}:q0-parent",
            f"step3:{replicate}:heuristic",
        )
    arm = cast(str, cell["candidate_arm"])
    shared = cell.get("shared_group")
    candidate = f"{arm}{index}-v2"
    opponent = cast(str, cell["opponent"])
    candidate_rng = (
        f"step3:{replicate}:{shared}:candidate"
        if isinstance(shared, str)
        else f"step3:{replicate}:{arm}{index}:candidate"
    )
    return candidate, opponent, candidate_rng, f"step3:{replicate}:{opponent}"


def local_arena_report(
    cell: Mapping[str, object], records: Sequence[object], retained: Mapping[str, object]
) -> dict[str, object]:
    """Reconstruct every semantic arena-report field without production arena aggregation."""
    a_id, b_id, a_rng, b_rng = expected_agent_id(cell)
    agents = retained.get("agents")
    if not isinstance(agents, Mapping):
        raise ValidationError("arena report agent metadata missing")
    a_config = cast(Mapping[str, object], agents["a"])["config"]
    b_config = cast(Mapping[str, object], agents["b"])["config"]
    if (
        cast(Mapping[str, object], agents["a"]).get("id") != a_id
        or cast(Mapping[str, object], agents["b"]).get("id") != b_id
    ):
        raise ValidationError("arena policy identifiers differ from frozen cell")
    pair_count = cast(int, cell["paired_blocks"])
    master = cast(int, cell["master_seed"])
    if len(records) != pair_count * 2:
        raise ValidationError("arena record cardinality differs")
    wins = {a_id: 0, b_id: 0}
    pair_wins: dict[str, int] = {}
    pair_games: dict[str, int] = {}
    seat_games = {PlayerId.PLAYER_ONE: 0, PlayerId.PLAYER_TWO: 0}
    seat_wins = {PlayerId.PLAYER_ONE: 0, PlayerId.PLAYER_TWO: 0}
    reasons: Counter[str] = Counter()
    margin = turns = decisions = 0
    for position, raw in enumerate(records):
        record = cast(object, raw)
        pair_index = position // 2
        a_first = position % 2 == 0
        pair_id = f"pair-{pair_index:06d}"
        pair_domain = f"arena:pair:{pair_index}"
        setup = derive_seed(master, f"{pair_domain}:setup") & ((1 << 64) - 1)
        seed_a = derive_seed(master, f"{pair_domain}:agent:{a_rng}")
        seed_b = derive_seed(master, f"{pair_domain}:agent:{b_rng}")
        expected_ids = (a_id, b_id) if a_first else (b_id, a_id)
        expected_seeds = (seed_a, seed_b) if a_first else (seed_b, seed_a)
        expected_domains = (
            (f"agent:{a_rng}", f"agent:{b_rng}")
            if a_first
            else (f"agent:{b_rng}", f"agent:{a_rng}")
        )
        if (
            record.run_id != cell["run_id"]
            or record.game_id != f"{pair_id}-{'a' if a_first else 'b'}-first"
            or record.pair_id != pair_id
            or record.replay.seed != setup
            or normalize_config(record.replay.config) != retained["game_config"]
        ):
            raise ValidationError("arena record schedule identity differs")
        if (
            tuple(seat.agent_id for seat in record.seats) != expected_ids
            or tuple(seat.seed for seat in record.seats) != expected_seeds
            or tuple(seat.rng_domain for seat in record.seats) != expected_domains
        ):
            raise ValidationError("arena record seat/RNG identity differs")
        expected_configs = (a_config, b_config) if a_first else (b_config, a_config)
        if tuple(dict(seat.config) for seat in record.seats) != expected_configs:
            raise ValidationError("arena record policy config differs")
        winner_id = record.seats[0 if record.winner is PlayerId.PLAYER_ONE else 1].agent_id
        if winner_id not in wins:
            raise ValidationError("arena winner agent differs")
        wins[winner_id] += 1
        pair_wins[pair_id] = pair_wins.get(pair_id, 0) + int(winner_id == a_id)
        pair_games[pair_id] = pair_games.get(pair_id, 0) + 1
        a_seat = PlayerId.PLAYER_ONE if record.seats[0].agent_id == a_id else PlayerId.PLAYER_TWO
        seat_games[a_seat] += 1
        seat_wins[a_seat] += int(winner_id == a_id)
        index = 0 if a_seat is PlayerId.PLAYER_ONE else 1
        margin += record.final_scores[index] - record.final_scores[1 - index]
        turns += record.turn_count
        decisions += record.decision_count
        reasons[record.terminal_reason] += 1
    if set(pair_games) != {f"pair-{index:06d}" for index in range(pair_count)} or any(
        value != 2 for value in pair_games.values()
    ):
        raise ValidationError("arena paired blocks differ")
    pair_values = tuple(pair_wins[f"pair-{index:06d}"] for index in range(pair_count))
    elapsed = number(retained.get("elapsed_seconds"), "retained elapsed")
    total = len(records)
    a_wins = wins[a_id]
    return {
        "arena_report_schema_version": 1,
        "run_id": cell["run_id"],
        "agents": {"a": {"id": a_id, "config": a_config}, "b": {"id": b_id, "config": b_config}},
        "total_games": total,
        "paired_seed_count": pair_count,
        "wins": {"a": a_wins, "b": wins[b_id]},
        "agent_a_win_rate": a_wins / total,
        "confidence_interval_95": list(local_wilson(a_wins, total)),
        "wilson_confidence_interval_95": list(local_wilson(a_wins, total)),
        "paired_seed_outcomes": [
            {"pair_id": f"pair-{index:06d}", "agent_a_wins": pair_values[index]}
            for index in range(pair_count)
        ],
        "paired_bootstrap_confidence_interval_95": local_pair_bootstrap(pair_values, master),
        "agent_a_by_seat": {
            seat.value: {
                "games": seat_games[seat],
                "wins": seat_wins[seat],
                "win_rate": seat_wins[seat] / seat_games[seat],
            }
            for seat in PlayerId
        },
        "terminal_reasons": dict(sorted(reasons.items())),
        "average_score_margin": margin / total,
        "average_turns": turns / total,
        "average_decisions": decisions / total,
        "elapsed_seconds": elapsed,
        "games_per_second": total / elapsed if elapsed else math.inf,
        "master_seed": master,
        "seed_derivation": SEED_DERIVATION,
        "rng_algorithm": RNG_ALGORITHM,
        "game_config": retained["game_config"],
        "rules_fingerprint": rules_fingerprint(),
        "code_fingerprint": code_fingerprint(),
    }


def pair_scores(report: Mapping[str, object]) -> tuple[float, ...]:
    rows = report.get("paired_seed_outcomes")
    if not isinstance(rows, list):
        raise ValidationError("arena has no paired outcomes")
    return tuple(
        number(cast(Mapping[str, object], row).get("agent_a_wins"), "pair wins") / 2 for row in rows
    )


def check_cross_cell_alignment(records_by_key: Mapping[str, Sequence[object]]) -> None:
    """Check record-level shared setup, physical seat, opponent config, and RNG identities."""

    def compare(
        left_key: str, right_key: str, opponent: str | None = None, *, all_rng: bool = True
    ) -> None:
        left = records_by_key[left_key]
        right = records_by_key[right_key]
        if len(left) != len(right):
            raise ValidationError("cross-cell record count differs")
        for first, second in zip(left, right, strict=True):
            if (first.replay.seed, first.pair_id, first.game_id) != (
                second.replay.seed,
                second.pair_id,
                second.game_id,
            ):
                raise ValidationError("cross-cell setup/pair/game identity differs")
            if all_rng and tuple((seat.seed, seat.rng_domain) for seat in first.seats) != tuple(
                (seat.seed, seat.rng_domain) for seat in second.seats
            ):
                raise ValidationError("cross-cell physical seat/RNG identity differs")
            if opponent is not None:
                one = next((seat for seat in first.seats if seat.agent_id == opponent), None)
                two = next((seat for seat in second.seats if seat.agent_id == opponent), None)
                if (
                    one is None
                    or two is None
                    or (dict(one.config), one.seed, one.rng_domain)
                    != (dict(two.config), two.seed, two.rng_domain)
                ):
                    raise ValidationError("shared opponent identity/config/RNG differs")

    for replicate in REPLICATES:
        index = replicate.removeprefix("replicate-")
        compare(f"C{index}-v2-vs-v1", f"M{index}-v2-vs-v1")
        for opponent in ("q0-parent", "random", "historical-q0", "q1", "q2", "q3", "q4"):
            compare(f"C{index}-v2-vs-{opponent}", f"M{index}-v2-vs-{opponent}", opponent)
        compare(f"C{index}-v2-vs-heuristic", f"M{index}-v2-vs-heuristic", "heuristic")
        compare(
            f"C{index}-v2-vs-heuristic",
            f"q0-parent-vs-heuristic-reference-{index}",
            "heuristic",
            all_rng=False,
        )
        compare(
            f"M{index}-v2-vs-heuristic",
            f"q0-parent-vs-heuristic-reference-{index}",
            "heuristic",
            all_rng=False,
        )


def nested(rows: tuple[tuple[float, ...], ...], domain: str) -> dict[str, object]:
    if len(rows) != 3 or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValidationError("bad independent nested rows")
    count = len(rows[0])
    seed = derive_seed(ROOT_SEED, f"step3:nested-bootstrap:{domain}") & ((1 << 63) - 1)
    rng = DeterministicRandom(seed, f"step3:nested-bootstrap:{domain}")
    values: list[float] = []
    for _ in range(RESAMPLES):
        selected = tuple(rng.randbelow(3) for _ in range(3))
        values.append(
            sum(
                sum(rows[index][rng.randbelow(count)] for _ in range(count)) / count
                for index in selected
            )
            / 3
        )
    values.sort()
    return {
        "point_estimate": sum(sum(row) / count for row in rows) / 3,
        "interval": [values[LOWER], values[UPPER]],
        "seed": seed,
        "domain": domain,
    }


def joint(
    control: tuple[tuple[float, ...], ...],
    mixed: tuple[tuple[float, ...], ...],
    *,
    interaction: bool,
) -> dict[str, object]:
    count = len(control[0]) if control else 0
    if (
        len(control) != 3
        or len(mixed) != 3
        or count == 0
        or any(len(row) != count for row in (*control, *mixed))
    ):
        raise ValidationError("independent joint architecture rows are malformed")
    domain = "interaction" if interaction else "pooled-architecture-main-effect"
    seed = derive_seed(ROOT_SEED, f"step3:nested-bootstrap:{domain}") & ((1 << 63) - 1)
    rng = DeterministicRandom(seed, f"step3:nested-bootstrap:{domain}")
    values: list[float] = []
    for _ in range(RESAMPLES):
        selected = tuple(rng.randbelow(3) for _ in range(3))
        means: list[float] = []
        for replicate in selected:
            indexes = tuple(rng.randbelow(count) for _ in range(count))
            c = sum(control[replicate][index] for index in indexes) / count
            m = sum(mixed[replicate][index] for index in indexes) / count
            means.append(m - c if interaction else (m + c) / 2)
        values.append(sum(means) / 3)
    values.sort()
    point_c = sum(sum(row) / count for row in control) / 3
    point_m = sum(sum(row) / count for row in mixed) / 3
    return {
        "point_estimate": point_m - point_c if interaction else (point_m + point_c) / 2,
        "interval": [values[LOWER], values[UPPER]],
        "seed": seed,
        "domain": domain,
    }


def macro(rows: Mapping[str, tuple[tuple[float, ...], ...]], arm: str) -> dict[str, object]:
    if set(rows) != set(OPPONENTS):
        raise ValidationError("macro opponent keys differ")
    domain = f"{arm}-equal-eight-opponent-macro"
    seed = derive_seed(ROOT_SEED, f"step3:nested-bootstrap:{domain}") & ((1 << 63) - 1)
    rng = DeterministicRandom(seed, f"step3:nested-bootstrap:{domain}")
    values: list[float] = []
    for resample_index in range(RESAMPLES):
        selected = tuple(rng.randbelow(3) for _ in range(3))
        replicate_means: list[float] = []
        for occurrence_index, replicate in enumerate(selected):
            opponents: list[float] = []
            for opponent in OPPONENTS:
                row = rows[opponent][replicate]
                inner = (
                    f"{domain}:opponent:{opponent}:resample:{resample_index}:"
                    f"occurrence:{occurrence_index}"
                )
                local = DeterministicRandom(
                    derive_seed(ROOT_SEED, f"step3:nested-bootstrap:{inner}") & ((1 << 63) - 1),
                    f"step3:nested-bootstrap:{inner}",
                )
                opponents.append(
                    sum(row[local.randbelow(len(row))] for _ in range(len(row))) / len(row)
                )
            replicate_means.append(sum(opponents) / len(opponents))
        values.append(sum(replicate_means) / 3)
    values.sort()
    point = sum(
        sum(sum(row) / len(row) for row in rows[opponent]) / 3 for opponent in OPPONENTS
    ) / len(OPPONENTS)
    return {
        "point_estimate": point,
        "interval": [values[LOWER], values[UPPER]],
        "seed": seed,
        "domain": domain,
    }


def same_block(left: Mapping[str, object], right: Mapping[str, object]) -> None:
    left_rows = cast(Mapping[str, object], left["report"]).get("paired_seed_outcomes")
    right_rows = cast(Mapping[str, object], right["report"]).get("paired_seed_outcomes")
    if (
        not isinstance(left_rows, list)
        or not isinstance(right_rows, list)
        or len(left_rows) != len(right_rows)
    ):
        raise ValidationError("shared arena block cardinality differs")
    if any(
        cast(Mapping[str, object], one).get("pair_id")
        != cast(Mapping[str, object], two).get("pair_id")
        for one, two in zip(left_rows, right_rows, strict=True)
    ):
        raise ValidationError("shared arena pair identifiers differ")


def local_statistics(arena_artifacts: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    architecture: dict[str, list[tuple[float, ...]]] = {arm: [] for arm in ARMS}
    data: list[tuple[float, ...]] = []
    parent: dict[str, list[tuple[float, ...]]] = {arm: [] for arm in ARMS}
    random_rows: dict[str, list[tuple[float, ...]]] = {arm: [] for arm in ARMS}
    heuristic: dict[str, list[tuple[float, ...]]] = {arm: [] for arm in ARMS}
    external: dict[str, dict[str, list[tuple[float, ...]]]] = {
        arm: {opponent: [] for opponent in OPPONENTS} for arm in ARMS
    }
    seats = {arm: 1.0 for arm in ARMS}
    per_replicate: list[dict[str, object]] = []
    for replicate in REPLICATES:
        index = replicate.removeprefix("replicate-")
        reference = arena_artifacts[f"q0-parent-vs-heuristic-reference-{index}"]
        row: dict[str, object] = {"replicate_id": replicate, "arms": {}}
        for arm in ARMS:
            direct = arena_artifacts[f"{arm}{index}-v2-vs-v1"]
            candidate_parent = arena_artifacts[f"{arm}{index}-v2-vs-q0-parent"]
            candidate_random = arena_artifacts[f"{arm}{index}-v2-vs-random"]
            candidate_heuristic = arena_artifacts[f"{arm}{index}-v2-vs-heuristic"]
            same_block(candidate_heuristic, reference)
            architecture[arm].append(pair_scores(cast(Mapping[str, object], direct["report"])))
            parent[arm].append(pair_scores(cast(Mapping[str, object], candidate_parent["report"])))
            random_rows[arm].append(
                pair_scores(cast(Mapping[str, object], candidate_random["report"]))
            )
            h = pair_scores(cast(Mapping[str, object], candidate_heuristic["report"]))
            ref = pair_scores(cast(Mapping[str, object], reference["report"]))
            heuristic[arm].append(tuple(a - b for a, b in zip(h, ref, strict=True)))
            for opponent in OPPONENTS:
                external[arm][opponent].append(
                    pair_scores(
                        cast(
                            Mapping[str, object],
                            arena_artifacts[f"{arm}{index}-v2-vs-{opponent}"]["report"],
                        )
                    )
                )
            candidate_seats: list[float] = []
            for artifact in (direct, candidate_parent, candidate_random, candidate_heuristic):
                report = cast(Mapping[str, object], artifact["report"])
                candidate_seats.extend(
                    number(cast(Mapping[str, object], item).get("win_rate"), "seat")
                    for item in cast(Mapping[str, object], report["agent_a_by_seat"]).values()
                )
            seats[arm] = min(seats[arm], min(candidate_seats))
            cast(dict[str, object], row["arms"])[arm] = {
                "architecture_v2_minus_v1": sum(architecture[arm][-1]) / len(architecture[arm][-1]),
                "v2_vs_q0_parent": sum(parent[arm][-1]) / len(parent[arm][-1]),
                "v2_vs_random": sum(random_rows[arm][-1]) / len(random_rows[arm][-1]),
                "v2_minus_q0_parent_heuristic": sum(heuristic[arm][-1]) / len(heuristic[arm][-1]),
            }
        data_row = pair_scores(
            cast(Mapping[str, object], arena_artifacts[f"M{index}-v2-vs-C{index}-v2"]["report"])
        )
        data.append(data_row)
        row["v2_data_effect_mixed_minus_control"] = sum(data_row) / len(data_row)
        per_replicate.append(row)
    nested_rows: dict[str, object] = {
        "control_architecture_v2_minus_v1": nested(
            tuple(architecture["C"]), "control-architecture-v2-minus-v1"
        ),
        "mixed_architecture_v2_minus_v1": nested(
            tuple(architecture["M"]), "mixed-architecture-v2-minus-v1"
        ),
        "pooled_architecture_main_effect": joint(
            tuple(architecture["C"]), tuple(architecture["M"]), interaction=False
        ),
        "v2_data_effect_mixed_minus_control": nested(
            tuple(data), "v2-data-effect-mixed-minus-control"
        ),
        "data_x_architecture_interaction": joint(
            tuple(architecture["C"]), tuple(architecture["M"]), interaction=True
        ),
    }
    for arm in ARMS:
        nested_rows[f"{arm}_v2_vs_q0_parent"] = nested(tuple(parent[arm]), f"{arm}-v2-vs-q0-parent")
        nested_rows[f"{arm}_v2_vs_random"] = nested(tuple(random_rows[arm]), f"{arm}-v2-vs-random")
        nested_rows[f"{arm}_v2_minus_q0_parent_heuristic"] = nested(
            tuple(heuristic[arm]), f"{arm}-v2-minus-parent-heuristic"
        )
        nested_rows[f"{arm}_equal_eight_opponent_macro"] = macro(
            {name: tuple(values) for name, values in external[arm].items()}, arm
        )
    return {
        "nested": nested_rows,
        "minimum_v2_candidate_seat": seats,
        "per_replicate": per_replicate,
    }


def local_selection(statistics: Mapping[str, object]) -> dict[str, object]:
    nested_rows = cast(Mapping[str, object], statistics["nested"])
    seats = cast(Mapping[str, object], statistics["minimum_v2_candidate_seat"])
    replicates = cast(list[object], statistics["per_replicate"])

    def lower(name: str) -> float:
        return number(cast(Mapping[str, object], nested_rows[name])["interval"][0], name)

    def point(name: str) -> float:
        return number(cast(Mapping[str, object], nested_rows[name])["point_estimate"], name)

    arms: dict[str, object] = {}
    viable: list[tuple[float, str]] = []
    for arm in ARMS:
        architecture_name = (
            "control_architecture_v2_minus_v1" if arm == "C" else "mixed_architecture_v2_minus_v1"
        )
        replicate_wins = sum(
            number(
                cast(Mapping[str, object], cast(Mapping[str, object], row["arms"])[arm])[
                    "architecture_v2_minus_v1"
                ],
                "architecture",
            )
            > 0.5
            for row in replicates
        )
        conditions = {
            "architecture_lower_strictly_above_50": lower(architecture_name) > 0.5,
            "two_of_three_architecture_replicates_above_50": replicate_wins >= 2,
            "parent_lower_above_50": lower(f"{arm}_v2_vs_q0_parent") > 0.5,
            "random_lower_above_50": lower(f"{arm}_v2_vs_random") > 0.5,
            "parent_heuristic_difference_lower_above_minus_5pp": lower(
                f"{arm}_v2_minus_q0_parent_heuristic"
            )
            > -0.05,
            "v2_candidate_seat_floor": number(seats[arm], "seat") >= 0.45,
            "tactical_invariants": True,
            "integrity": True,
        }
        floor = min(
            lower(architecture_name) - 0.5,
            lower(f"{arm}_v2_vs_q0_parent") - 0.5,
            lower(f"{arm}_v2_vs_random") - 0.5,
            lower(f"{arm}_v2_minus_q0_parent_heuristic") + 0.05,
            number(seats[arm], "seat") - 0.45,
        )
        if all(conditions[key] for key in ("tactical_invariants", "integrity")):
            viable.append((floor, arm))
        arms[arm] = {
            "conditions": conditions,
            "advancing_structured_recipe": all(conditions.values()),
            "robustness_floor": floor,
            "equal_opponent_macro": point(f"{arm}_equal_eight_opponent_macro"),
            "architecture_point": point(architecture_name),
        }
    advancing = [
        arm
        for arm in ARMS
        if cast(Mapping[str, object], arms[arm])["advancing_structured_recipe"] is True
    ]
    pooled = lower("pooled_architecture_main_effect") > 0.5
    if len(advancing) == 2 and pooled:
        classification = "general_advancement"
    elif len(advancing) == 1:
        classification = "data_dependent_recipe_advance"
    else:
        crosses = any(
            lower(name)
            <= 0.5
            <= number(cast(Mapping[str, object], nested_rows[name])["interval"][1], name)
            for name in (
                "control_architecture_v2_minus_v1",
                "mixed_architecture_v2_minus_v1",
            )
        )
        classification = "inconclusive_does_not_advance" if crosses else "does_not_advance"
    selected = None
    if viable:
        selected = sorted(
            viable,
            key=lambda row: (
                row[0],
                number(cast(Mapping[str, object], arms[row[1]])["equal_opponent_macro"], "macro"),
                number(
                    cast(Mapping[str, object], arms[row[1]])["architecture_point"], "architecture"
                ),
                1 if row[1] == "C" else 0,
            ),
            reverse=True,
        )[0][1]
    return {"classification": classification, "development_selected_step4_input_recipe": selected}


def current_source() -> dict[str, object]:
    identity = inspect_source_identity()
    root = repository_root()
    status = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all", "--ignored=no"),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return {
        **identity.to_data(),
        "rules_fingerprint": rules_fingerprint(),
        "code_fingerprint": code_fingerprint(),
        "runner_clean_check": {
            "version": "structured-model-v2-runner-clean-source-v1",
            "tracked_and_nonignored_untracked_clean": not bool(status.strip()),
            "status_sha256": hashlib.sha256(status).hexdigest(),
        },
    }


def check_completed_cardinality(
    output: Path, plan: Mapping[str, object], result: Mapping[str, object]
) -> None:
    state = read(output / "execution-state.json")
    if state.get("completed") is not True:
        raise ValidationError("result execution state is not complete")
    cells = cast(Mapping[str, object], plan["arenas"]).get("cells")
    if not isinstance(cells, list) or len(cells) != 60:
        raise ValidationError("plan does not retain all 60 Step-3 arenas")
    keys = {f"{arm}{index}" for arm in ARMS for index in ("1", "2", "3")}
    for name in ("datasets", "checkpoints"):
        value = result.get(name)
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValidationError(f"result {name} cardinality differs")
    arenas = result.get("arenas")
    expected_arenas = {
        cast(str, cast(Mapping[str, object], cell)["key"])
        for cell in cells
        if isinstance(cell, Mapping)
    }
    if not isinstance(arenas, Mapping) or set(arenas) != expected_arenas:
        raise ValidationError("result arena cardinality differs")
    for name in ("statistics", "safety-report", "selection", "runtime-extrapolation", "checksums"):
        artifact = read(output / f"{name}.json")
        check_artifact(artifact, name)
    for key in keys:
        artifact = read(output / "datasets" / key / "audit.json")
        check_artifact(artifact, "dataset")
        if cast(Mapping[str, object], result["datasets"]).get(key) != artifact.get(
            "artifact_fingerprint"
        ):
            raise ValidationError("result dataset reference differs")
    checksums = read(output / "checksums.json")
    files = checksums.get("files")
    if not isinstance(files, Mapping):
        raise ValidationError("checksum files mapping is malformed")
    for relative, expected in files.items():
        path = output / cast(str, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValidationError("checksum payload differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    source_guard()
    output = args.output.resolve()
    plan = read(output / "plan.json")
    result = read(output / "result.json")
    if plan.get("plan_fingerprint") != digest(
        {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    ):
        raise ValidationError("plan fingerprint mismatch")
    execution = cast(Mapping[str, object], plan["execution"])
    source = current_source()
    if execution.get("claim_eligible") is True and plan.get("source") != source:
        raise ValidationError("claim validator requires exact clean source, lock, code, and rules")
    clean = source.get("runner_clean_check")
    if (
        not isinstance(clean, Mapping)
        or clean.get("tracked_and_nonignored_untracked_clean") is not True
    ):
        raise ValidationError("validator requires a clean current source tree")
    if result.get("result_fingerprint") != digest(
        {key: value for key, value in result.items() if key != "result_fingerprint"}
    ):
        raise ValidationError("result fingerprint mismatch")
    check_completed_cardinality(output, plan, result)
    if execution.get("claim_eligible") is not True and not args.allow_smoke:
        raise ValidationError("bounded evidence requires --allow-smoke")
    started = time.perf_counter()
    dataset_started = time.perf_counter()
    inputs = cast(list[object], plan["inputs"])
    q0 = load_checkpoint(Path(cast(str, plan["q0_path"])))
    plan_cells = {
        cast(str, cast(Mapping[str, object], cell)["key"]): cast(Mapping[str, object], cell)
        for cell in cast(list[object], cast(Mapping[str, object], plan["arenas"])["cells"])
        if isinstance(cell, Mapping)
    }
    arena_artifacts: dict[str, Mapping[str, object]] = {}
    arena_records: dict[str, Sequence[object]] = {}
    for item in inputs:
        entry = cast(Mapping[str, object], item)
        key = cast(str, entry["key"])
        corpus = Path(cast(str, plan["step2_root"])).parents[1] / cast(
            str, cast(Mapping[str, object], entry["corpus"])["path"]
        )
        v1 = Path(cast(str, plan["step2_root"])).parents[1] / cast(
            str, cast(Mapping[str, object], entry["v1_dataset"])["path"]
        )
        replay_dataset(corpus, v1, output / "datasets" / key / "dataset.npz")
        checkpoint = load_structured_checkpoint(output / "checkpoints" / key)
        if (
            cast(Mapping[str, object], result["checkpoints"]).get(key)
            != checkpoint.checkpoint_fingerprint
        ):
            raise ValidationError("result checkpoint reference differs")
        sources = checkpoint.manifest.get("sources")
        parent = sources.get("q0_parent") if isinstance(sources, Mapping) else None
        if not isinstance(parent, Mapping) or parent.get("tensor_digest") != tensor_digest(
            q0.model.state_dict()
        ):
            raise ValidationError("structured checkpoint q0 lineage differs")
    arena_started = time.perf_counter()
    for report_path in sorted((output / "arenas").glob("*/report.json")):
        artifact = read(report_path)
        check_artifact(artifact, "arena")
        key = cast(str, cast(Mapping[str, object], artifact["cell"])["key"])
        _records_manifest, records = load_corpus(report_path.parent / "records", verify_code=False)
        local_tactical(records)
        report = cast(Mapping[str, object], artifact["report"])
        local_report = local_arena_report(plan_cells[key], records, report)
        if local_report != report:
            raise ValidationError("locally recomputed complete arena report differs")
        arena_records[key] = records
        local_artifact = dict(artifact)
        local_artifact["report"] = local_report
        arena_artifacts[key] = local_artifact
    check_cross_cell_alignment(arena_records)
    statistics_started = time.perf_counter()
    stored = read(output / "statistics.json")
    check_artifact(stored, "statistics")
    independent = local_statistics(arena_artifacts)
    nested_stored = cast(Mapping[str, object], stored["nested"])
    for key, value in cast(Mapping[str, object], independent["nested"]).items():
        actual = cast(Mapping[str, object], nested_stored[key])
        expected = cast(Mapping[str, object], value)
        if actual.get("point_estimate") != expected.get("point_estimate") or actual.get(
            "interval"
        ) != expected.get("interval"):
            raise ValidationError(f"independent nested statistic differs: {key}")
    safety = read(output / "safety-report.json")
    check_artifact(safety, "safety")
    selection = read(output / "selection.json")
    check_artifact(selection, "selection")
    independent_selection = local_selection(independent)
    retained_selection = {
        key: value
        for key, value in selection.items()
        if key not in {"artifact_fingerprint", "plan_fingerprint", "version"}
    }
    if (
        result.get("selection") != {"version": selection.get("version"), **retained_selection}
        or retained_selection.get("classification") != independent_selection["classification"]
        or retained_selection.get("development_selected_step4_input_recipe")
        != independent_selection["development_selected_step4_input_recipe"]
    ):
        raise ValidationError("independent selection differs")
    elapsed = time.perf_counter() - started
    cells = cast(list[object], cast(Mapping[str, object], plan["arenas"])["cells"])
    smoke_pairs = min(
        cast(int, cast(Mapping[str, object], cell)["paired_blocks"])
        for cell in cells
        if isinstance(cell, Mapping)
    )
    arena_seconds = statistics_started - arena_started
    statistics_seconds = time.perf_counter() - statistics_started
    dataset_seconds = arena_started - dataset_started
    arena_multiplier = 30_000 / max(2 * 60 * smoke_pairs, 1)
    full_bootstrap_blocks = 2 * 500 + 500 + 2 * 500 + 2 * 200 + 2 * 300 + 2 * 1600 + 2 * 500
    smoke_bootstrap_blocks = (
        2 * smoke_pairs
        + smoke_pairs
        + 2 * smoke_pairs
        + 2 * smoke_pairs
        + 2 * smoke_pairs
        + 16 * smoke_pairs
        + 2 * smoke_pairs
    )
    statistics_multiplier = full_bootstrap_blocks / smoke_bootstrap_blocks
    validation_runtime = {
        "version": "m7-structured-model-v2-validation-runtime-v2",
        "validation_seconds": elapsed,
        "observed_phase_seconds": {
            "dataset_feature_reconstruction": dataset_seconds,
            "arena_aggregate_and_tactical_replay": arena_seconds,
            "nested_statistics_and_selection": statistics_seconds,
        },
        "projected_full_phase_seconds": {
            "dataset_feature_reconstruction": dataset_seconds,
            "arena_aggregate_and_tactical_replay": arena_seconds * arena_multiplier,
            "nested_statistics_and_selection": statistics_seconds * statistics_multiplier,
        },
        "projected_full_seconds": dataset_seconds
        + arena_seconds * arena_multiplier
        + statistics_seconds * statistics_multiplier,
        "arena_multiplier": arena_multiplier,
        "statistics_multiplier": statistics_multiplier,
        "dataset_reconstruction_is_full_corpus_in_smoke": True,
    }
    validation_runtime["artifact_fingerprint"] = digest(validation_runtime)
    runtime_path = output / "validation-runtime.json"
    if runtime_path.exists() and read(runtime_path) != validation_runtime:
        raise ValidationError("existing validation runtime differs")
    runtime_path.write_bytes(canonical(validation_runtime) + b"\n")
    validation = {
        "version": "m7-structured-model-v2-independent-validation-v1",
        "status": "passed",
        "plan_fingerprint": plan["plan_fingerprint"],
        "checks": {
            "source_guard": True,
            "local_519_feature_reconstruction": True,
            "dataset_rows_splits_labels": True,
            "checkpoint_q0_lineage": True,
            "local_arena_aggregation_and_tactical_replay": True,
            "nested_statistics": True,
            "selection_artifact": True,
        },
        "artifact_fingerprint": "",
    }
    validation["artifact_fingerprint"] = digest(
        {key: value for key, value in validation.items() if key != "artifact_fingerprint"}
    )
    if (output / "validation.json").exists() and read(output / "validation.json") != validation:
        raise ValidationError("existing independent validation differs")
    (output / "validation.json").write_bytes(canonical(validation) + b"\n")
    print(json.dumps(validation, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"structured v2 validation failed: {exc}", file=sys.stderr)
        raise
