"""Immutable authoritative domain model for the rules engine."""

from dataclasses import dataclass
from enum import StrEnum

from .cards import CANONICAL_DECK, CardName


class PlayerId(StrEnum):
    PLAYER_ONE = "player_one"
    PLAYER_TWO = "player_two"

    def other(self) -> "PlayerId":
        """Return the opposing player."""
        return PlayerId.PLAYER_TWO if self is PlayerId.PLAYER_ONE else PlayerId.PLAYER_ONE


class Phase(StrEnum):
    PLAY = "play"
    RECRUIT = "recruit"
    END = "end"
    TERMINAL = "terminal"


class OfferSlot(StrEnum):
    FACE_UP = "face_up"
    FACE_DOWN = "face_down"


class OutcomeReason(StrEnum):
    CONDITION = "condition"
    DECK_EXHAUSTION = "deck_exhaustion"


class OutcomeResolution(StrEnum):
    SOLE_CANDIDATE = "sole_candidate"
    ACTIVE_CONDITION_TIE = "active_condition_tie"
    HIGH_SCORE = "high_score"
    ACTIVE_SCORE_TIE = "active_score_tie"


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Normalized setup configuration.

    Milestone 1 deliberately validates against the canonical base deck while retaining the deck in
    the configuration so replays fully describe their setup.
    """

    starting_player: PlayerId = PlayerId.PLAYER_ONE
    deck: tuple[CardName, ...] = CANONICAL_DECK


@dataclass(frozen=True, slots=True)
class Offer:
    """The two cards awaiting the opponent's recruit choice."""

    offered_by: PlayerId
    face_up: CardName
    face_down: CardName


@dataclass(frozen=True, slots=True)
class PlayOfferAction:
    """Play an ordered face-up/face-down pair by card name."""

    revision: int
    actor: PlayerId
    face_up: CardName
    face_down: CardName


@dataclass(frozen=True, slots=True)
class RecruitAction:
    """Recruit one stable offer slot."""

    revision: int
    actor: PlayerId
    slot: OfferSlot


type Action = PlayOfferAction | RecruitAction


@dataclass(frozen=True, slots=True)
class PlayDecision:
    revision: int
    actor: PlayerId
    turn: int


@dataclass(frozen=True, slots=True)
class RecruitDecision:
    revision: int
    actor: PlayerId
    turn: int
    face_up: CardName


type Decision = PlayDecision | RecruitDecision


@dataclass(frozen=True, slots=True)
class CompletedTurn:
    """Public record of a resolved offer and its assignments."""

    turn: int
    active_player: PlayerId
    face_up: CardName
    face_down: CardName
    chosen_slot: OfferSlot
    opponent_recruited: CardName
    active_recruited: CardName
    score_changes: tuple[int, int]


@dataclass(frozen=True, slots=True)
class TerminalFacts:
    """All material facts used to adjudicate a terminal result."""

    scores: tuple[int, int]
    codebreaker_counts: tuple[int, int]
    daredevil_counts: tuple[int, int]
    score_gap_winners: tuple[PlayerId, ...]
    instant_winners: tuple[PlayerId, ...]
    instant_losers: tuple[PlayerId, ...]
    candidate_winners: tuple[PlayerId, ...]
    deck_empty: bool
    next_player_hand_size: int


@dataclass(frozen=True, slots=True)
class TerminalOutcome:
    winner: PlayerId
    reason: OutcomeReason
    resolution: OutcomeResolution
    active_player: PlayerId
    turn: int
    facts: TerminalFacts


@dataclass(frozen=True, slots=True)
class GameState:
    """Complete trusted game state. Never serialize this object to an untrusted viewer."""

    config: GameConfig
    seed: int
    deck: tuple[CardName, ...]
    hands: tuple[tuple[CardName, ...], tuple[CardName, ...]]
    recruited: tuple[tuple[CardName, ...], tuple[CardName, ...]]
    scores: tuple[int, int]
    active_player: PlayerId
    turn: int
    phase: Phase
    revision: int
    offer: Offer | None = None
    actions: tuple[Action, ...] = ()
    history: tuple[CompletedTurn, ...] = ()
    outcome: TerminalOutcome | None = None


def player_index(player: PlayerId) -> int:
    """Return the stable tuple index for a player."""
    return 0 if player is PlayerId.PLAYER_ONE else 1
