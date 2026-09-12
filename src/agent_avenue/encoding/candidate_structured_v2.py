"""Safe public-history candidate encoder for the structured value model.

This module consumes a player-visible observation and a semantic legal action only.  It deliberately
uses the pure public-material terminal evaluator instead of creating an authoritative successor.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from agent_avenue.engine.cards import CARD_DEFINITIONS, CardEffect, CardName, recruit_effect
from agent_avenue.engine.model import (
    Action,
    CompletedTurn,
    OfferSlot,
    PlayerId,
    PlayOfferAction,
    RecruitAction,
    TerminalOutcome,
)
from agent_avenue.engine.terminal import TERMINAL_EVALUATOR_VERSION, adjudicate_position
from agent_avenue.observation.model import (
    PlayerObservation,
    PublicPlayer,
    RecruitContext,
)

from .candidate_v1 import (
    ENCODER_FINGERPRINT as V1_ENCODER_FINGERPRINT,
)
from .candidate_v1 import (
    ENCODER_VERSION as V1_ENCODER_VERSION,
)
from .candidate_v1 import (
    FEATURE_NAMES as V1_FEATURE_NAMES,
)
from .candidate_v1 import (
    encode_candidate as encode_v1_candidate,
)
from .candidate_v1 import (
    information_visible_unseen_counts,
)
from .schema import CARD_ORDER

ENCODER_VERSION: Final[str] = "candidate-public-structured-v2"
FEATURE_WIDTH: Final[int] = 519
V1_PREFIX_WIDTH: Final[int] = 87
HISTORY_WIDTH: Final[int] = 314
PLAY_CONSEQUENCE_WIDTH: Final[int] = 64
RECRUIT_CONSEQUENCE_WIDTH: Final[int] = 54
HISTORY_WINDOW: Final[int] = 8
HISTORY_EVENT_WIDTH: Final[int] = 39

_EFFECT_KINDS: Final[tuple[str, str, str]] = ("score", "win", "lose")


@dataclass(frozen=True, slots=True)
class StructuredCandidateEncoderConfig:
    """Normalization and semantic constants that define the v2 feature contract."""

    version: str = ENCODER_VERSION
    card_order: tuple[CardName, ...] = CARD_ORDER
    v1_encoder_version: str = V1_ENCODER_VERSION
    v1_encoder_fingerprint: str = V1_ENCODER_FINGERPRINT
    history_window: int = HISTORY_WINDOW
    history_turn_scale: int = 19
    history_score_change_min: int = -3
    history_score_change_max: int = 6
    history_score_change_scale: int = 6
    hand_scale: int = 4
    canonical_deck_size: int = 38
    terminal_evaluator_version: str = TERMINAL_EVALUATOR_VERSION
    play_consequence_version: str = "public-material-play-branches-v1"
    recruit_consequence_version: str = "safe-unseen-support-envelope-v1"

    def __post_init__(self) -> None:
        if self.version != ENCODER_VERSION:
            raise ValueError("unsupported structured encoder version")
        if self.card_order != CARD_ORDER:
            raise ValueError("structured encoder requires the canonical card order")
        if self.v1_encoder_version != V1_ENCODER_VERSION:
            raise ValueError("structured encoder requires candidate-public-v1")
        if self.v1_encoder_fingerprint != V1_ENCODER_FINGERPRINT:
            raise ValueError("structured encoder requires the current v1 prefix fingerprint")
        if self.history_window != HISTORY_WINDOW:
            raise ValueError("structured encoder requires an eight-turn history window")
        if self.history_turn_scale != 19 or self.hand_scale != 4 or self.canonical_deck_size != 38:
            raise ValueError("structured encoder normalization constants are fixed")
        if (
            self.history_score_change_min != -3
            or self.history_score_change_max != 6
            or self.history_score_change_scale != 6
        ):
            raise ValueError("structured history score normalization is fixed")
        if self.terminal_evaluator_version != TERMINAL_EVALUATOR_VERSION:
            raise ValueError("structured encoder requires the current terminal evaluator")
        if self.play_consequence_version != "public-material-play-branches-v1":
            raise ValueError("unsupported play consequence definition")
        if self.recruit_consequence_version != "safe-unseen-support-envelope-v1":
            raise ValueError("unsupported recruit consequence definition")

    def to_data(self) -> dict[str, object]:
        return {
            "canonical_deck_size": self.canonical_deck_size,
            "card_order": [card.value for card in self.card_order],
            "hand_scale": self.hand_scale,
            "history_score_change_max": self.history_score_change_max,
            "history_score_change_min": self.history_score_change_min,
            "history_score_change_scale": self.history_score_change_scale,
            "history_turn_scale": self.history_turn_scale,
            "history_window": self.history_window,
            "play_consequence_version": self.play_consequence_version,
            "recruit_consequence_version": self.recruit_consequence_version,
            "terminal_evaluator_version": self.terminal_evaluator_version,
            "v1_encoder_fingerprint": self.v1_encoder_fingerprint,
            "v1_encoder_version": self.v1_encoder_version,
            "version": self.version,
        }


def _history_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for event_index in range(HISTORY_WINDOW):
        prefix = f"history_{event_index}"
        names.append(f"{prefix}_valid")
        names.extend((f"{prefix}_active_player_self", f"{prefix}_active_player_opponent"))
        names.extend(f"{prefix}_face_up_{card.value}" for card in CARD_ORDER)
        names.extend(f"{prefix}_face_down_{card.value}" for card in CARD_ORDER)
        names.extend((f"{prefix}_chosen_face_up", f"{prefix}_chosen_face_down"))
        names.extend(f"{prefix}_opponent_recruited_{card.value}" for card in CARD_ORDER)
        names.extend(f"{prefix}_active_recruited_{card.value}" for card in CARD_ORDER)
        names.extend((f"{prefix}_self_score_change", f"{prefix}_opponent_score_change"))
    names.extend(("history_length_fraction", "history_truncated_fraction"))
    return tuple(names)


def _play_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for branch in ("recruit_face_up", "recruit_face_down"):
        prefix = f"play_{branch}"
        names.extend(f"{prefix}_self_card_{card.value}" for card in CARD_ORDER)
        names.extend(f"{prefix}_opponent_card_{card.value}" for card in CARD_ORDER)
        names.extend(f"{prefix}_self_effect_{kind}" for kind in _EFFECT_KINDS)
        names.extend(f"{prefix}_opponent_effect_{kind}" for kind in _EFFECT_KINDS)
        names.extend((f"{prefix}_self_terminal_win", f"{prefix}_self_terminal_loss"))
        names.extend((f"{prefix}_opponent_terminal_win", f"{prefix}_opponent_terminal_loss"))
        names.extend((f"{prefix}_self_score_delta", f"{prefix}_opponent_score_delta"))
        names.extend((f"{prefix}_self_resulting_count", f"{prefix}_opponent_resulting_count"))
    names.extend(
        (
            "play_post_play_hand_without_refill_fraction",
            "play_post_play_refill_count_fraction",
            "play_post_play_hand_size_fraction",
            "play_offer_same_card",
        )
    )
    return tuple(names)


def _recruit_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    names.extend(f"recruit_known_card_to_self_{card.value}" for card in CARD_ORDER)
    names.extend(f"recruit_known_card_to_opponent_{card.value}" for card in CARD_ORDER)
    names.extend(("recruit_hidden_to_self", "recruit_hidden_to_opponent"))
    names.extend(f"recruit_known_effect_self_{kind}" for kind in _EFFECT_KINDS)
    names.extend(f"recruit_known_effect_opponent_{kind}" for kind in _EFFECT_KINDS)
    names.extend(("recruit_guaranteed_terminal_self_win", "recruit_guaranteed_terminal_self_loss"))
    names.extend(
        ("recruit_guaranteed_terminal_opponent_win", "recruit_guaranteed_terminal_opponent_loss")
    )
    names.extend(("recruit_known_score_delta_self", "recruit_known_score_delta_opponent"))
    names.extend(("recruit_known_resulting_count_self", "recruit_known_resulting_count_opponent"))
    names.append("recruit_known_card_recipient_prior_count")
    names.extend(f"recruit_hidden_effect_support_self_{kind}" for kind in _EFFECT_KINDS)
    names.extend(f"recruit_hidden_effect_support_opponent_{kind}" for kind in _EFFECT_KINDS)
    names.extend(("recruit_hidden_score_min_self", "recruit_hidden_score_max_self"))
    names.extend(("recruit_hidden_score_min_opponent", "recruit_hidden_score_max_opponent"))
    names.extend(("recruit_hidden_card_mass_self", "recruit_hidden_card_mass_opponent"))
    names.extend(f"recruit_hidden_effect_support_bit_self_{kind}" for kind in _EFFECT_KINDS)
    names.extend(f"recruit_hidden_effect_support_bit_opponent_{kind}" for kind in _EFFECT_KINDS)
    names.append("recruit_safe_unseen_pool_fraction")
    names.extend(
        (
            "recruit_candidate_terminal_support_self_win",
            "recruit_candidate_terminal_support_self_loss",
        )
    )
    return tuple(names)


HISTORY_FEATURE_NAMES: Final[tuple[str, ...]] = _history_feature_names()
PLAY_CONSEQUENCE_FEATURE_NAMES: Final[tuple[str, ...]] = _play_feature_names()
RECRUIT_CONSEQUENCE_FEATURE_NAMES: Final[tuple[str, ...]] = _recruit_feature_names()
if len(HISTORY_FEATURE_NAMES) != HISTORY_WIDTH:  # pragma: no cover - schema guard
    raise RuntimeError("structured history width is not 314")
if len(PLAY_CONSEQUENCE_FEATURE_NAMES) != PLAY_CONSEQUENCE_WIDTH:  # pragma: no cover
    raise RuntimeError("structured play consequence width is not 64")
if len(RECRUIT_CONSEQUENCE_FEATURE_NAMES) != RECRUIT_CONSEQUENCE_WIDTH:  # pragma: no cover
    raise RuntimeError("structured recruit consequence width is not 54")

# Prefix names remain byte-for-byte and order-for-order the v1 contract.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    V1_FEATURE_NAMES
    + HISTORY_FEATURE_NAMES
    + PLAY_CONSEQUENCE_FEATURE_NAMES
    + RECRUIT_CONSEQUENCE_FEATURE_NAMES
)
if (
    len(FEATURE_NAMES) != FEATURE_WIDTH or len(set(FEATURE_NAMES)) != FEATURE_WIDTH
):  # pragma: no cover
    raise RuntimeError("structured feature schema is not a unique 519-value layout")

CONFIG: Final[StructuredCandidateEncoderConfig] = StructuredCandidateEncoderConfig()
ENCODER_CONFIG: Final[StructuredCandidateEncoderConfig] = CONFIG


def _traceability() -> Mapping[str, str]:
    sources: dict[str, str] = {}
    for name in V1_FEATURE_NAMES:
        sources[name] = "candidate-public-v1-prefix"
    for name in HISTORY_FEATURE_NAMES:
        sources[name] = "PlayerObservation.history.completed_public_turn"
    for name in PLAY_CONSEQUENCE_FEATURE_NAMES:
        sources[name] = "PlayerObservation.public_fields+legal_play_action+adjudicate_position"
    for name in RECRUIT_CONSEQUENCE_FEATURE_NAMES:
        sources[name] = "PlayerObservation.public_fields+safe_unseen_counts+adjudicate_position"
    return MappingProxyType(sources)


FEATURE_SOURCES: Final[Mapping[str, str]] = _traceability()
FEATURE_SOURCE_TRACEABILITY: Final[Mapping[str, str]] = FEATURE_SOURCES


@dataclass(frozen=True, slots=True)
class StructuredCandidateFeatureSchema:
    """Named v2 layout and every fixed normalization/consequence definition."""

    config: StructuredCandidateEncoderConfig
    feature_names: tuple[str, ...]
    feature_sources: Mapping[str, str]

    @property
    def version(self) -> str:
        return self.config.version

    @property
    def width(self) -> int:
        return len(self.feature_names)

    def to_data(self) -> dict[str, object]:
        return {
            "config": self.config.to_data(),
            "feature_names": list(self.feature_names),
            "feature_sources": dict(self.feature_sources),
            "version": self.version,
            "width": self.width,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


FEATURE_SCHEMA: Final[StructuredCandidateFeatureSchema] = StructuredCandidateFeatureSchema(
    CONFIG, FEATURE_NAMES, FEATURE_SOURCES
)
ENCODER_FINGERPRINT: Final[str] = FEATURE_SCHEMA.fingerprint
FEATURE_FINGERPRINT: Final[str] = ENCODER_FINGERPRINT


@dataclass(frozen=True, slots=True)
class StructuredCandidateEncoding:
    """Immutable named v2 vector with its non-negotiable schema identity."""

    vector: tuple[float, ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    version: str = ENCODER_VERSION
    fingerprint: str = ENCODER_FINGERPRINT

    def __post_init__(self) -> None:
        if type(self.vector) is not tuple or len(self.vector) != FEATURE_WIDTH:
            raise ValueError(
                f"structured candidate vector must contain exactly {FEATURE_WIDTH} values"
            )
        if any(type(value) is not float or not math.isfinite(value) for value in self.vector):
            raise ValueError("structured candidate vector values must be finite floats")
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("structured feature names do not match candidate-public-structured-v2")
        if self.version != ENCODER_VERSION or self.fingerprint != ENCODER_FINGERPRINT:
            raise ValueError(
                "structured encoding metadata does not match candidate-public-structured-v2"
            )

    @property
    def values(self) -> tuple[float, ...]:
        return self.vector

    @property
    def features(self) -> tuple[float, ...]:
        return self.vector

    @property
    def width(self) -> int:
        return len(self.vector)


# Compatibility spelling for callers that use the v1 type name.
CandidateEncoding = StructuredCandidateEncoding


def _f32(value: float) -> float:
    result = struct.unpack("!f", struct.pack("!f", value))[0]
    if not math.isfinite(result):
        raise ValueError("structured feature calculation produced a non-finite float32")
    return float(result)


def _one_hot(card: CardName | None) -> list[float]:
    result = [0.0] * len(CARD_ORDER)
    if card is not None:
        result[CARD_ORDER.index(card)] = 1.0
    return result


def _effect_one_hot(effect: CardEffect | None) -> list[float]:
    result = [0.0, 0.0, 0.0]
    if effect is not None:
        result[_EFFECT_KINDS.index(effect.kind)] = 1.0
    return result


def _score_delta(effect: CardEffect) -> int:
    return effect.points if effect.kind == "score" else 0


def _public_player(observation: PlayerObservation, player: PlayerId) -> PublicPlayer:
    matches = tuple(item for item in observation.players if item.player is player)
    if len(matches) != 1:
        raise ValueError("observation must contain exactly one public record for each player")
    return matches[0]


def _validate_history_item(item: CompletedTurn) -> None:
    if type(item) is not CompletedTurn:
        raise ValueError("history must contain CompletedTurn values")
    if not isinstance(item.active_player, PlayerId):
        raise ValueError("completed turn active player is invalid")
    if any(
        not isinstance(card, CardName)
        for card in (
            item.face_up,
            item.face_down,
            item.opponent_recruited,
            item.active_recruited,
        )
    ):
        raise ValueError("completed turn contains an invalid card")
    if item.chosen_slot not in (OfferSlot.FACE_UP, OfferSlot.FACE_DOWN):
        raise ValueError("completed turn chosen slot is invalid")
    if (
        type(item.turn) is not int
        or not 1 <= item.turn <= CONFIG.history_turn_scale
        or type(item.score_changes) is not tuple
        or len(item.score_changes) != 2
        or any(type(change) is not int for change in item.score_changes)
    ):
        raise ValueError("completed turn public fields are malformed")
    expected_opponent = item.face_up if item.chosen_slot is OfferSlot.FACE_UP else item.face_down
    expected_active = item.face_down if item.chosen_slot is OfferSlot.FACE_UP else item.face_up
    if (
        item.opponent_recruited is not expected_opponent
        or item.active_recruited is not expected_active
    ):
        raise ValueError("completed turn recruit assignments do not match its chosen slot")


def _history_features(observation: PlayerObservation) -> list[float]:
    history = observation.history
    if type(history) is not tuple or len(history) > CONFIG.history_turn_scale:
        raise ValueError(
            "history must be a public tuple no longer than the mechanical turn ceiling"
        )
    for item in history:
        _validate_history_item(item)
    recent = history[-HISTORY_WINDOW:]
    values: list[float] = [0.0] * ((HISTORY_WINDOW - len(recent)) * HISTORY_EVENT_WIDTH)
    for item in recent:
        self_change = item.score_changes[0 if observation.viewer is PlayerId.PLAYER_ONE else 1]
        opponent_change = item.score_changes[1 if observation.viewer is PlayerId.PLAYER_ONE else 0]
        values.append(1.0)
        values.extend(
            (
                1.0 if item.active_player is observation.viewer else 0.0,
                1.0 if item.active_player is observation.viewer.other() else 0.0,
            )
        )
        values.extend(_one_hot(item.face_up))
        values.extend(_one_hot(item.face_down))
        values.extend(
            (
                1.0 if item.chosen_slot is OfferSlot.FACE_UP else 0.0,
                1.0 if item.chosen_slot is OfferSlot.FACE_DOWN else 0.0,
            )
        )
        values.extend(_one_hot(item.opponent_recruited))
        values.extend(_one_hot(item.active_recruited))
        values.extend(
            (
                _f32(
                    max(
                        CONFIG.history_score_change_min,
                        min(CONFIG.history_score_change_max, self_change),
                    )
                    / CONFIG.history_score_change_scale
                ),
                _f32(
                    max(
                        CONFIG.history_score_change_min,
                        min(CONFIG.history_score_change_max, opponent_change),
                    )
                    / CONFIG.history_score_change_scale
                ),
            )
        )
    values.extend(
        (
            _f32(len(history) / CONFIG.history_turn_scale),
            _f32(max(len(history) - HISTORY_WINDOW, 0) / CONFIG.history_turn_scale),
        )
    )
    if len(values) != HISTORY_WIDTH:  # pragma: no cover - defensive schema guard
        raise RuntimeError("history feature calculation did not produce 314 values")
    return values


def _effect_for(player: PublicPlayer, card: CardName) -> tuple[CardEffect, int, int]:
    prior = player.recruited.count(card)
    resulting = prior + 1
    return recruit_effect(card, resulting), prior, resulting


def _with_card(
    observation: PlayerObservation,
    self_card: CardName,
    opponent_card: CardName,
) -> tuple[
    tuple[tuple[CardName, ...], tuple[CardName, ...]], tuple[int, int], CardEffect, CardEffect
]:
    self_player = _public_player(observation, observation.viewer)
    opponent_player = _public_player(observation, observation.viewer.other())
    self_effect, _, _ = _effect_for(self_player, self_card)
    opponent_effect, _, _ = _effect_for(opponent_player, opponent_card)
    if observation.viewer is PlayerId.PLAYER_ONE:
        recruited = (
            (*self_player.recruited, self_card),
            (*opponent_player.recruited, opponent_card),
        )
        scores = (
            self_player.score + _score_delta(self_effect),
            opponent_player.score + _score_delta(opponent_effect),
        )
    else:
        recruited = (
            (*opponent_player.recruited, opponent_card),
            (*self_player.recruited, self_card),
        )
        scores = (
            opponent_player.score + _score_delta(opponent_effect),
            self_player.score + _score_delta(self_effect),
        )
    return recruited, scores, self_effect, opponent_effect


def _terminal_for_cards(
    observation: PlayerObservation,
    self_card: CardName,
    opponent_card: CardName,
    *,
    deck_empty: bool,
    next_player_hand_size: int,
) -> tuple[TerminalOutcome | None, CardEffect, CardEffect]:
    recruited, scores, self_effect, opponent_effect = _with_card(
        observation, self_card, opponent_card
    )
    outcome = adjudicate_position(
        scores=scores,
        recruited=recruited,
        active_player=observation.active_player,
        turn=observation.turn,
        deck_empty=deck_empty,
        next_player_hand_size=next_player_hand_size,
    )
    return outcome, self_effect, opponent_effect


def _terminal_bits(
    outcome: TerminalOutcome | None, viewer: PlayerId
) -> tuple[list[float], list[float]]:
    if outcome is None:
        return [0.0, 0.0], [0.0, 0.0]
    # ``adjudicate_position`` is the only producer accepted here. Keeping the terminal data narrow
    # avoids coupling this safe feature builder to an authoritative state representation.
    winner = outcome.winner
    self_bits = [1.0, 0.0] if winner is viewer else [0.0, 1.0]
    opponent_bits = [0.0, 1.0] if winner is viewer else [1.0, 0.0]
    return self_bits, opponent_bits


def _play_branch_features(
    observation: PlayerObservation,
    self_card: CardName,
    opponent_card: CardName,
    *,
    deck_empty: bool,
    next_player_hand_size: int,
) -> list[float]:
    self_player = _public_player(observation, observation.viewer)
    opponent_player = _public_player(observation, observation.viewer.other())
    outcome, self_effect, opponent_effect = _terminal_for_cards(
        observation,
        self_card,
        opponent_card,
        deck_empty=deck_empty,
        next_player_hand_size=next_player_hand_size,
    )
    self_terminal, opponent_terminal = _terminal_bits(outcome, observation.viewer)
    _, _, self_count = _effect_for(self_player, self_card)
    _, _, opponent_count = _effect_for(opponent_player, opponent_card)
    values = _one_hot(self_card) + _one_hot(opponent_card)
    values.extend(_effect_one_hot(self_effect))
    values.extend(_effect_one_hot(opponent_effect))
    values.extend(self_terminal)
    values.extend(opponent_terminal)
    values.extend(
        (
            _f32(_score_delta(self_effect) / 6),
            _f32(_score_delta(opponent_effect) / 6),
            _f32(self_count / CARD_DEFINITIONS[self_card].copies),
            _f32(opponent_count / CARD_DEFINITIONS[opponent_card].copies),
        )
    )
    if len(values) != 30:  # pragma: no cover - defensive schema guard
        raise RuntimeError("play consequence branch did not produce 30 values")
    return values


def _play_consequence_features(
    observation: PlayerObservation, action: PlayOfferAction
) -> list[float]:
    self_player = _public_player(observation, observation.viewer)
    opponent_player = _public_player(observation, observation.viewer.other())
    hand_after_offer = self_player.hand_size - 2
    refill = min(CONFIG.hand_scale - hand_after_offer, observation.remaining_deck_count)
    post_deck_empty = observation.remaining_deck_count - refill == 0
    values = _play_branch_features(
        observation,
        action.face_down,
        action.face_up,
        deck_empty=post_deck_empty,
        next_player_hand_size=opponent_player.hand_size,
    )
    values.extend(
        _play_branch_features(
            observation,
            action.face_up,
            action.face_down,
            deck_empty=post_deck_empty,
            next_player_hand_size=opponent_player.hand_size,
        )
    )
    values.extend(
        (
            _f32(hand_after_offer / CONFIG.hand_scale),
            _f32(refill / CONFIG.hand_scale),
            _f32((hand_after_offer + refill) / CONFIG.hand_scale),
            1.0 if action.face_up is action.face_down else 0.0,
        )
    )
    if len(values) != PLAY_CONSEQUENCE_WIDTH:  # pragma: no cover
        raise RuntimeError("play consequence calculation did not produce 64 values")
    return values


def _hidden_support(
    observation: PlayerObservation,
    action: RecruitAction,
    known_card: CardName,
) -> list[float]:
    self_player = _public_player(observation, observation.viewer)
    opponent_player = _public_player(observation, observation.viewer.other())
    self_known = action.slot is OfferSlot.FACE_UP
    known_recipient = self_player if self_known else opponent_player
    known_effect, known_prior, known_count = _effect_for(known_recipient, known_card)
    unseen_counts = information_visible_unseen_counts(observation)
    support = tuple(
        (card, count) for card, count in zip(CARD_ORDER, unseen_counts, strict=True) if count > 0
    )
    if not support:
        raise ValueError("recruit observation must have a public-consistent hidden-card support")
    unseen_total = sum(count for _, count in support)

    values = _one_hot(known_card if self_known else None)
    values.extend(_one_hot(known_card if not self_known else None))
    values.extend((0.0 if self_known else 1.0, 1.0 if self_known else 0.0))
    values.extend(_effect_one_hot(known_effect if self_known else None))
    values.extend(_effect_one_hot(known_effect if not self_known else None))

    outcome_for_hidden: list[tuple[CardName, int, TerminalOutcome | None, CardEffect]] = []
    for hidden_card, multiplicity in support:
        self_card = known_card if self_known else hidden_card
        opponent_card = hidden_card if self_known else known_card
        outcome, _, _ = _terminal_for_cards(
            observation,
            self_card,
            opponent_card,
            deck_empty=observation.remaining_deck_count == 0,
            next_player_hand_size=self_player.hand_size,
        )
        hidden_recipient = self_player if not self_known else opponent_player
        hidden_effect, _, _ = _effect_for(hidden_recipient, hidden_card)
        outcome_for_hidden.append((hidden_card, multiplicity, outcome, hidden_effect))

    self_results = [
        _terminal_bits(outcome, observation.viewer)[0] for _, _, outcome, _ in outcome_for_hidden
    ]
    opponent_results = [
        _terminal_bits(outcome, observation.viewer)[1] for _, _, outcome, _ in outcome_for_hidden
    ]
    values.extend(
        (
            1.0 if all(bits == [1.0, 0.0] for bits in self_results) else 0.0,
            1.0 if all(bits == [0.0, 1.0] for bits in self_results) else 0.0,
            1.0 if all(bits == [1.0, 0.0] for bits in opponent_results) else 0.0,
            1.0 if all(bits == [0.0, 1.0] for bits in opponent_results) else 0.0,
        )
    )
    values.extend(
        (
            _f32(_score_delta(known_effect) / 6) if self_known else 0.0,
            _f32(_score_delta(known_effect) / 6) if not self_known else 0.0,
            _f32(known_count / CARD_DEFINITIONS[known_card].copies) if self_known else 0.0,
            _f32(known_count / CARD_DEFINITIONS[known_card].copies) if not self_known else 0.0,
            _f32(known_prior / CARD_DEFINITIONS[known_card].copies),
        )
    )

    hidden_effect_mass = {kind: 0 for kind in _EFFECT_KINDS}
    hidden_effect_bits = {kind: 0.0 for kind in _EFFECT_KINDS}
    hidden_scores: list[int] = []
    for _, multiplicity, _, effect in outcome_for_hidden:
        hidden_effect_mass[effect.kind] += multiplicity
        hidden_effect_bits[effect.kind] = 1.0
        hidden_scores.append(_score_delta(effect))
    support_fractions = [_f32(hidden_effect_mass[kind] / unseen_total) for kind in _EFFECT_KINDS]
    score_min = _f32(min(hidden_scores) / 6)
    score_max = _f32(max(hidden_scores) / 6)
    if self_known:
        values.extend([0.0, 0.0, 0.0])
        values.extend(support_fractions)
        values.extend((0.0, 0.0, score_min, score_max))
        values.extend((0.0, _f32(1 / max(unseen_total, 1))))
        values.extend([0.0, 0.0, 0.0])
        values.extend(hidden_effect_bits[kind] for kind in _EFFECT_KINDS)
    else:
        values.extend(support_fractions)
        values.extend([0.0, 0.0, 0.0])
        values.extend((score_min, score_max, 0.0, 0.0))
        values.extend((_f32(1 / max(unseen_total, 1)), 0.0))
        values.extend(hidden_effect_bits[kind] for kind in _EFFECT_KINDS)
        values.extend([0.0, 0.0, 0.0])
    values.extend(
        (
            _f32(unseen_total / CONFIG.canonical_deck_size),
            1.0 if any(bits == [1.0, 0.0] for bits in self_results) else 0.0,
            1.0 if any(bits == [0.0, 1.0] for bits in self_results) else 0.0,
        )
    )
    if len(values) != RECRUIT_CONSEQUENCE_WIDTH:  # pragma: no cover
        raise RuntimeError("recruit consequence calculation did not produce 54 values")
    return values


def _inactive(width: int) -> list[float]:
    return [0.0] * width


def encode_candidate(observation: PlayerObservation, action: Action) -> StructuredCandidateEncoding:
    """Encode one legal candidate without consulting hidden state or successor transitions."""
    # The v1 call supplies the literal prefix and centralizes existing legal-action/public
    # observation validation without changing that encoder's behavior.
    prefix = encode_v1_candidate(observation, action).vector
    if len(prefix) != V1_PREFIX_WIDTH:  # pragma: no cover
        raise RuntimeError("candidate-public-v1 prefix width changed")
    history = _history_features(observation)
    if isinstance(action, PlayOfferAction):
        play = _play_consequence_features(observation, action)
        recruit = _inactive(RECRUIT_CONSEQUENCE_WIDTH)
    elif isinstance(action, RecruitAction):
        if not isinstance(observation.decision, RecruitContext):  # v1 has already validated this
            raise ValueError("recruit action requires a public recruit context")
        play = _inactive(PLAY_CONSEQUENCE_WIDTH)
        recruit = _hidden_support(observation, action, observation.decision.face_up)
    else:  # pragma: no cover - v1 validation restricts Action members
        raise ValueError("unsupported structured candidate action")
    vector = prefix + tuple(history) + tuple(play) + tuple(recruit)
    if len(vector) != FEATURE_WIDTH:  # pragma: no cover
        raise RuntimeError("structured candidate encoder did not produce 519 values")
    return StructuredCandidateEncoding(vector)


def encode(observation: PlayerObservation, action: Action) -> StructuredCandidateEncoding:
    """Short alias for :func:`encode_candidate`."""
    return encode_candidate(observation, action)


__all__ = [
    "CONFIG",
    "ENCODER_CONFIG",
    "ENCODER_FINGERPRINT",
    "ENCODER_VERSION",
    "FEATURE_FINGERPRINT",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA",
    "FEATURE_SOURCES",
    "FEATURE_SOURCE_TRACEABILITY",
    "FEATURE_WIDTH",
    "HISTORY_EVENT_WIDTH",
    "HISTORY_FEATURE_NAMES",
    "HISTORY_WIDTH",
    "HISTORY_WINDOW",
    "PLAY_CONSEQUENCE_FEATURE_NAMES",
    "PLAY_CONSEQUENCE_WIDTH",
    "RECRUIT_CONSEQUENCE_FEATURE_NAMES",
    "RECRUIT_CONSEQUENCE_WIDTH",
    "V1_PREFIX_WIDTH",
    "CandidateEncoding",
    "StructuredCandidateEncoderConfig",
    "StructuredCandidateEncoding",
    "StructuredCandidateFeatureSchema",
    "encode",
    "encode_candidate",
]
