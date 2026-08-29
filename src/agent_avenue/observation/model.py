"""Immutable player-visible observation types."""

from dataclasses import dataclass

from agent_avenue.engine.cards import CardName
from agent_avenue.engine.model import (
    Action,
    CompletedTurn,
    OfferSlot,
    Phase,
    PlayerId,
    TerminalOutcome,
)


@dataclass(frozen=True, slots=True)
class PublicPlayer:
    player: PlayerId
    score: int
    recruited: tuple[CardName, ...]
    hand_size: int


@dataclass(frozen=True, slots=True)
class PlayContext:
    kind: str
    revision: int
    actor: PlayerId


@dataclass(frozen=True, slots=True)
class RecruitContext:
    kind: str
    revision: int
    actor: PlayerId
    offered_by: PlayerId
    face_up: CardName
    known_face_down: CardName | None
    slots: tuple[OfferSlot, OfferSlot] = (OfferSlot.FACE_UP, OfferSlot.FACE_DOWN)


@dataclass(frozen=True, slots=True)
class TerminalContext:
    kind: str
    outcome: TerminalOutcome


type ObservationDecision = PlayContext | RecruitContext | TerminalContext


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    """Allowlisted safe view. It deliberately has no deck order or opposing hand field."""

    viewer: PlayerId
    own_hand: tuple[CardName, ...]
    players: tuple[PublicPlayer, PublicPlayer]
    active_player: PlayerId
    turn: int
    phase: Phase
    remaining_deck_count: int
    history: tuple[CompletedTurn, ...]
    decision: ObservationDecision
    legal_actions: tuple[Action, ...]
