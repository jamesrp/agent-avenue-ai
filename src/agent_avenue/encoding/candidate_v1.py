"""The candidate-public-v1 action-conditioned encoder.

Only a :class:`PlayerObservation` and one semantic legal action are accepted.  In particular, this
module deliberately has no function that accepts ``GameState`` or an authoritative successor.
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Final

from agent_avenue.engine.cards import CARD_DEFINITIONS, CardName
from agent_avenue.engine.model import (
    Action,
    OfferSlot,
    Phase,
    PlayerId,
    PlayOfferAction,
    RecruitAction,
)
from agent_avenue.observation.model import (
    PlayContext,
    PlayerObservation,
    PublicPlayer,
    RecruitContext,
)

from .schema import CARD_ORDER, CandidateEncoderConfig, CandidateFeatureSchema

ENCODER_VERSION: Final[str] = "candidate-public-v1"
CONFIG: Final[CandidateEncoderConfig] = CandidateEncoderConfig()

_REPEATABLE_SCORE_CARDS: Final[tuple[CardName, ...]] = (
    CardName.DOUBLE_AGENT,
    CardName.ENFORCER,
    CardName.SABOTEUR,
    CardName.SENTINEL,
)
_THRESHOLD_FEATURES: Final[tuple[tuple[CardName, tuple[int, ...]], ...]] = (
    *((card, (1, 2, 3)) for card in _REPEATABLE_SCORE_CARDS),
    (CardName.CODEBREAKER, (1, 2)),
    (CardName.DAREDEVIL, (1, 2)),
    (CardName.SIDEKICK, (1,)),
    (CardName.MOLE, (1,)),
)


def _feature_names() -> tuple[str, ...]:
    names: list[str] = [
        "phase_play",
        "phase_recruit",
        "turn_fraction",
        "remaining_deck_fraction",
        "self_score",
        "opponent_score",
        "score_gap",
        "self_hand_size",
        "opponent_hand_size",
    ]
    names.extend(f"self_hand_count_{card.value}" for card in CARD_ORDER)
    for viewpoint in ("self", "opponent"):
        names.extend(
            f"{viewpoint}_recruited_{card.value}_at_least_{threshold}"
            for card, thresholds in _THRESHOLD_FEATURES
            for threshold in thresholds
        )
    names.extend(f"unseen_count_{card.value}" for card in CARD_ORDER)
    names.extend(f"current_recruit_face_up_{card.value}" for card in CARD_ORDER)
    names.extend(f"candidate_play_face_up_{card.value}" for card in CARD_ORDER)
    names.extend(f"candidate_play_face_down_{card.value}" for card in CARD_ORDER)
    names.extend(("candidate_recruit_face_up", "candidate_recruit_face_down"))
    return tuple(names)


FEATURE_NAMES: Final[tuple[str, ...]] = _feature_names()
FEATURE_WIDTH: Final[int] = len(FEATURE_NAMES)
if FEATURE_WIDTH != 87:  # pragma: no cover - import-time contract guard
    raise RuntimeError(f"candidate-public-v1 must have 87 features, got {FEATURE_WIDTH}")

FEATURE_SCHEMA: Final[CandidateFeatureSchema] = CandidateFeatureSchema(CONFIG, FEATURE_NAMES)
ENCODER_FINGERPRINT: Final[str] = FEATURE_SCHEMA.fingerprint
FEATURE_FINGERPRINT: Final[str] = ENCODER_FINGERPRINT
ENCODER_CONFIG: Final[CandidateEncoderConfig] = CONFIG

__all__ = [
    "CARD_ORDER",
    "CONFIG",
    "ENCODER_CONFIG",
    "ENCODER_FINGERPRINT",
    "ENCODER_VERSION",
    "FEATURE_FINGERPRINT",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA",
    "FEATURE_WIDTH",
    "CandidateEncoding",
    "encode",
    "encode_candidate",
    "information_visible_unseen_counts",
]


@dataclass(frozen=True, slots=True)
class CandidateEncoding:
    """A named, immutable feature vector produced by the v1 encoder."""

    vector: tuple[float, ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    version: str = ENCODER_VERSION
    fingerprint: str = ENCODER_FINGERPRINT

    def __post_init__(self) -> None:
        if type(self.vector) is not tuple or len(self.vector) != FEATURE_WIDTH:
            raise ValueError(f"candidate vector must contain exactly {FEATURE_WIDTH} values")
        if any(type(value) is not float or not math.isfinite(value) for value in self.vector):
            raise ValueError("candidate vector values must be finite floats")
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("candidate feature names do not match candidate-public-v1")
        if self.version != ENCODER_VERSION or self.fingerprint != ENCODER_FINGERPRINT:
            raise ValueError("candidate encoding metadata does not match candidate-public-v1")

    @property
    def values(self) -> tuple[float, ...]:
        """Alias used by callers that call vectors ``values``."""
        return self.vector

    @property
    def features(self) -> tuple[float, ...]:
        """Alias for integrations that call model inputs ``features``."""
        return self.vector

    @property
    def width(self) -> int:
        return len(self.vector)


def _require_exact_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _public_player(observation: PlayerObservation, player: PlayerId) -> PublicPlayer:
    matches = tuple(item for item in observation.players if item.player is player)
    if len(matches) != 1:
        raise ValueError("observation must contain exactly one public record for each player")
    return matches[0]


def _validate_card_tuple(cards: object, field: str) -> tuple[CardName, ...]:
    if type(cards) is not tuple or any(not isinstance(card, CardName) for card in cards):
        raise ValueError(f"{field} must be a tuple of CardName values")
    counts = Counter(cards)
    if any(count > CARD_DEFINITIONS[card].copies for card, count in counts.items()):
        raise ValueError(f"{field} contains more copies of a card than the canonical deck")
    return cards


def _validate_observation(
    observation: PlayerObservation, action: Action
) -> tuple[PublicPlayer, PublicPlayer]:
    if type(observation) is not PlayerObservation:
        raise TypeError("encoder requires a PlayerObservation")
    if not isinstance(observation.viewer, PlayerId):
        raise ValueError("observation.viewer must be a PlayerId")
    own_hand = _validate_card_tuple(observation.own_hand, "own_hand")
    if type(observation.players) is not tuple or len(observation.players) != 2:
        raise ValueError("observation.players must contain two public players")
    if any(type(player) is not PublicPlayer for player in observation.players):
        raise ValueError("observation.players must contain PublicPlayer values")
    if {item.player for item in observation.players} != set(PlayerId):
        raise ValueError("observation.players must contain both distinct players")
    self_player = _public_player(observation, observation.viewer)
    opponent_player = _public_player(observation, observation.viewer.other())
    for item in (self_player, opponent_player):
        _require_exact_int(item.score, "public score")
        _require_exact_int(item.hand_size, "public hand size", minimum=0)
        if item.hand_size > CONFIG.hand_scale:
            raise ValueError("public hand size cannot exceed four")
        recruited_counts = Counter(item.recruited)
        impossible_threshold = {
            CardName.CODEBREAKER: 2,
            CardName.DAREDEVIL: 2,
            CardName.SIDEKICK: 1,
            CardName.MOLE: 1,
        }
        if any(recruited_counts[card] > maximum for card, maximum in impossible_threshold.items()):
            raise ValueError("recruited counts contain an impossible nonterminal threshold")
    if self_player.hand_size != len(own_hand):
        raise ValueError("self public hand_size must match own_hand")
    if not isinstance(observation.active_player, PlayerId):
        raise ValueError("observation.active_player must be a PlayerId")
    if not isinstance(observation.phase, Phase) or observation.phase in (Phase.END, Phase.TERMINAL):
        raise ValueError("encoder requires a nonterminal play or recruit observation")
    turn = _require_exact_int(observation.turn, "turn", minimum=1)
    if turn > CONFIG.turn_scale:
        raise ValueError("turn exceeds the mechanical 19-turn ceiling")
    remaining_deck = _require_exact_int(
        observation.remaining_deck_count, "remaining_deck_count", minimum=0
    )
    if remaining_deck > CONFIG.deck_scale:
        raise ValueError("remaining_deck_count exceeds the post-deal deck")
    if type(observation.legal_actions) is not tuple or not observation.legal_actions:
        raise ValueError("observation.legal_actions must be a non-empty tuple")
    if any(
        type(item) not in (PlayOfferAction, RecruitAction) for item in observation.legal_actions
    ):
        raise ValueError("legal_actions contains an unsupported action type")
    if action not in observation.legal_actions:
        raise ValueError("candidate action must be a member of observation.legal_actions")
    if len(set(observation.legal_actions)) != len(observation.legal_actions):
        raise ValueError("observation.legal_actions must not contain duplicates")
    decision = observation.decision
    if not isinstance(decision, PlayContext | RecruitContext):
        raise ValueError("encoder requires a play or recruit decision context")
    if decision.actor is not observation.viewer:
        raise ValueError("observation must be owned by its decision actor")
    if decision.kind not in ("play", "recruit"):
        raise ValueError("decision context has an unsupported kind")
    _require_exact_int(decision.revision, "decision revision", minimum=0)
    for legal_action in observation.legal_actions:
        if legal_action.revision != decision.revision or legal_action.actor is not decision.actor:
            raise ValueError("legal actions must match the decision revision and actor")
    if decision.revision != action.revision:
        raise ValueError("candidate revision does not match the decision")
    if action.actor is not decision.actor:
        raise ValueError("candidate actor does not match the decision actor")
    if isinstance(decision, PlayContext):
        if observation.phase is not Phase.PLAY or decision.kind != "play":
            raise ValueError("play context must describe a play phase")
        if decision.actor is not observation.active_player:
            raise ValueError("play actor must be the active player")
        if not isinstance(action, PlayOfferAction):
            raise ValueError("play context requires a PlayOfferAction")
        for legal_action in observation.legal_actions:
            if not isinstance(legal_action, PlayOfferAction):
                raise ValueError("play context legal actions must all be play actions")
            if legal_action.face_up not in own_hand or legal_action.face_down not in own_hand:
                raise ValueError("play candidate cards must be in own_hand")
            if legal_action.face_up is legal_action.face_down and (
                own_hand.count(legal_action.face_up) < 2 or len(set(own_hand)) != 1
            ):
                raise ValueError("play action contains an illegal duplicate-card offer")
        if action.face_up not in own_hand or action.face_down not in own_hand:
            raise ValueError("play candidate cards must be in own_hand")
    else:
        if observation.phase is not Phase.RECRUIT or decision.kind != "recruit":
            raise ValueError("recruit context must describe a recruit phase")
        if decision.actor is not observation.active_player.other():
            raise ValueError("recruit actor must be the non-active player")
        if decision.offered_by is not observation.active_player:
            raise ValueError("recruit offer must be owned by the active player")
        if decision.known_face_down is not None:
            raise ValueError("recruit actor cannot know the face-down card")
        if not isinstance(decision.face_up, CardName):
            raise ValueError("recruit face-up card must be a CardName")
        if type(decision.slots) is not tuple or decision.slots != (
            OfferSlot.FACE_UP,
            OfferSlot.FACE_DOWN,
        ):
            raise ValueError("recruit slots must be the canonical face-up/face-down pair")
        if not isinstance(action, RecruitAction):
            raise ValueError("recruit context requires a RecruitAction")
        if any(
            not isinstance(legal_action, RecruitAction)
            or legal_action.slot not in (OfferSlot.FACE_UP, OfferSlot.FACE_DOWN)
            for legal_action in observation.legal_actions
        ):
            raise ValueError("recruit context legal actions must be recruit slots")
    # Recruited and own-hand cards are public/actor-visible.  This check also makes the unknown
    # multiset calculation reject forged observations rather than silently clipping negatives.
    visible = Counter(own_hand)
    visible.update(self_player.recruited)
    visible.update(opponent_player.recruited)
    if isinstance(decision, RecruitContext):
        visible[decision.face_up] += 1
    if any(visible[card] > CARD_DEFINITIONS[card].copies for card in CARD_ORDER):
        raise ValueError("observation contains impossible public card counts")
    unknown_total = sum(CARD_DEFINITIONS[card].copies - visible[card] for card in CARD_ORDER)
    expected_unknown = remaining_deck + opponent_player.hand_size
    if isinstance(decision, RecruitContext):
        expected_unknown += 1
    if unknown_total != expected_unknown:
        raise ValueError("observation card counts do not match remaining deck and hand sizes")
    return self_player, opponent_player


def information_visible_unseen_counts(observation: PlayerObservation) -> tuple[int, ...]:
    """Return unknown card counts derivable from the supplied safe observation.

    The result combines the opponent hand, deck, and (during recruit) the hidden face-down offer;
    it never attempts to identify any of those cards.
    """
    # A harmless legal candidate is needed to use the same strict observation checks.  This function
    # is public for diagnostics, so validate the observation without requiring a caller to duplicate
    # the candidate selection contract.
    if type(observation) is not PlayerObservation:
        raise TypeError("observation must be a PlayerObservation")
    if not observation.legal_actions:
        raise ValueError("observation must contain legal actions")
    self_player, opponent_player = _validate_observation(observation, observation.legal_actions[0])
    visible = Counter(observation.own_hand)
    visible.update(self_player.recruited)
    visible.update(opponent_player.recruited)
    if isinstance(observation.decision, RecruitContext):
        visible[observation.decision.face_up] += 1
    return tuple(CARD_DEFINITIONS[card].copies - visible[card] for card in CARD_ORDER)


def _clip_fraction(value: int, limit: int, scale: int) -> float:
    return float(max(-limit, min(limit, value)) / scale)


def _one_hot(card: CardName | None, names: list[float]) -> None:
    if card is not None:
        names[CARD_ORDER.index(card)] = 1.0


def encode_candidate(observation: PlayerObservation, action: Action) -> CandidateEncoding:
    """Encode one legal semantic action from a nonterminal safe observation.

    The returned tuple is viewpoint-relative: the two public player records are always represented
    as self and opponent, regardless of whether the viewer is Player One or Player Two.
    """
    self_player, opponent_player = _validate_observation(observation, action)
    decision = observation.decision
    assert isinstance(decision, PlayContext | RecruitContext)
    values: list[float] = [
        1.0 if isinstance(decision, PlayContext) else 0.0,
        1.0 if isinstance(decision, RecruitContext) else 0.0,
        float(observation.turn / CONFIG.turn_scale),
        float(observation.remaining_deck_count / CONFIG.deck_scale),
        _clip_fraction(self_player.score, CONFIG.score_limit, CONFIG.score_limit),
        _clip_fraction(opponent_player.score, CONFIG.score_limit, CONFIG.score_limit),
        _clip_fraction(
            self_player.score - opponent_player.score, CONFIG.gap_limit, CONFIG.gap_limit
        ),
        float(self_player.hand_size / CONFIG.hand_scale),
        float(opponent_player.hand_size / CONFIG.hand_scale),
    ]
    own_counts = Counter(observation.own_hand)
    values.extend(float(own_counts[card] / CARD_DEFINITIONS[card].copies) for card in CARD_ORDER)
    for recruited in (self_player.recruited, opponent_player.recruited):
        counts = Counter(recruited)
        values.extend(
            float(counts[card] >= threshold)
            for card, thresholds in _THRESHOLD_FEATURES
            for threshold in thresholds
        )
    unseen = information_visible_unseen_counts(observation)
    values.extend(
        float(count / CARD_DEFINITIONS[card].copies)
        for card, count in zip(CARD_ORDER, unseen, strict=True)
    )

    current_face_up = decision.face_up if isinstance(decision, RecruitContext) else None
    one_hot_values: list[float] = [0.0] * len(CARD_ORDER)
    _one_hot(current_face_up, one_hot_values)
    values.extend(one_hot_values)
    play_up = action.face_up if isinstance(action, PlayOfferAction) else None
    play_down = action.face_down if isinstance(action, PlayOfferAction) else None
    one_hot_values = [0.0] * len(CARD_ORDER)
    _one_hot(play_up, one_hot_values)
    values.extend(one_hot_values)
    one_hot_values = [0.0] * len(CARD_ORDER)
    _one_hot(play_down, one_hot_values)
    values.extend(one_hot_values)
    values.extend(
        (
            1.0 if isinstance(action, RecruitAction) and action.slot is OfferSlot.FACE_UP else 0.0,
            1.0
            if isinstance(action, RecruitAction) and action.slot is OfferSlot.FACE_DOWN
            else 0.0,
        )
    )
    if len(values) != FEATURE_WIDTH:  # pragma: no cover - schema guard
        raise RuntimeError(
            f"candidate encoder produced {len(values)} values, expected {FEATURE_WIDTH}"
        )
    return CandidateEncoding(tuple(values))


# Short alias for callers that treat the encoder as a function.
def encode(observation: PlayerObservation, action: Action) -> CandidateEncoding:
    """Alias for :func:`encode_candidate`."""
    return encode_candidate(observation, action)
