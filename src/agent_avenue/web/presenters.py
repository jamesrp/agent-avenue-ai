"""Allowlisted conversion of safe observations to template view models."""

import hashlib
import json
from collections import Counter
from typing import TypedDict

from agent_avenue.engine import CARD_DEFINITIONS, CardName, PlayerId
from agent_avenue.engine.model import CompletedTurn, TerminalOutcome
from agent_avenue.observation.model import PlayerObservation, RecruitContext

CARD_LABELS = {card: card.value.replace("_", " ").title() for card in CardName}
PLAYER_LABELS = {
    PlayerId.PLAYER_ONE: "Player One",
    PlayerId.PLAYER_TWO: "Player Two",
}


class CardView(TypedDict):
    value: str
    label: str
    effect: str
    alt: str
    image: str


def card_view(card: CardName) -> CardView:
    definition = CARD_DEFINITIONS[card]
    effects = []
    for card_effect in definition.effects:
        effects.append(
            card_effect.kind.upper() if card_effect.kind != "score" else f"{card_effect.points:+d}"
        )
    effect = " / ".join(effects)
    label = CARD_LABELS[card]
    return {
        "value": card.value,
        "label": label,
        "effect": effect,
        "alt": f"{label} {effect}",
        "image": f"cards/{card.value.replace('_', '-')}.png",
    }


def recruited_views(cards: tuple[CardName, ...]) -> list[dict[str, object]]:
    counts = Counter(cards)
    return [
        {**card_view(card), "count": count}
        for card, count in sorted(counts.items(), key=lambda item: item[0].value)
    ]


def observation_view(observation: PlayerObservation) -> dict[str, object]:
    """Present only fields already available in a player-safe observation."""
    players = [
        {
            "label": PLAYER_LABELS[player.player],
            "score": player.score,
            "hand_size": player.hand_size,
            "recruited": recruited_views(player.recruited),
        }
        for player in observation.players
    ]
    decision = observation.decision
    offer: dict[str, object] | None = None
    if isinstance(decision, RecruitContext):
        offer = {
            "face_up": card_view(decision.face_up),
            "face_down": (
                None if decision.known_face_down is None else card_view(decision.known_face_down)
            ),
        }
    return {
        "viewer": PLAYER_LABELS[observation.viewer],
        "active_player": PLAYER_LABELS[observation.active_player],
        "turn": observation.turn,
        "phase": observation.phase.value.title(),
        "remaining_deck_count": observation.remaining_deck_count,
        "own_hand": [card_view(card) for card in observation.own_hand],
        "allow_matching_offer": len(set(observation.own_hand)) == 1,
        "players": players,
        "decision": decision,
        "offer": offer,
        "history": [completed_turn_view(item) for item in observation.history],
    }


def completed_turn_view(item: CompletedTurn) -> dict[str, object]:
    return {
        "turn": item.turn,
        "active_player": PLAYER_LABELS[item.active_player],
        "face_up": CARD_LABELS[item.face_up],
        "face_down": CARD_LABELS[item.face_down],
        "chooser": PLAYER_LABELS[item.active_player.other()],
        "chosen_slot": item.chosen_slot.value.replace("_", " ").title(),
        "opponent_recruited": CARD_LABELS[item.opponent_recruited],
        "active_recruited": CARD_LABELS[item.active_recruited],
        "score_changes": item.score_changes,
    }


def outcome_view(outcome: TerminalOutcome) -> dict[str, object]:
    return {
        "winner": PLAYER_LABELS[outcome.winner],
        "reason": outcome.reason.value.replace("_", " ").title(),
        "resolution": outcome.resolution.value.replace("_", " ").title(),
        "active_player": PLAYER_LABELS[outcome.active_player],
        "turn": outcome.turn,
    }


def public_board_view(observation: PlayerObservation) -> dict[str, object]:
    """Present viewer-neutral public state without either hand."""
    view = observation_view(observation)
    return {
        "active_player": view["active_player"],
        "turn": view["turn"],
        "phase": view["phase"],
        "remaining_deck_count": view["remaining_deck_count"],
        "players": view["players"],
        "history": view["history"],
    }


def public_fingerprint(seed: int, history: tuple[CompletedTurn, ...]) -> str:
    """Fingerprint reproducible public events, never authoritative hidden state."""
    value = {
        "seed": seed,
        "history": [
            [
                item.turn,
                item.active_player.value,
                item.face_up.value,
                item.face_down.value,
                item.chosen_slot.value,
                list(item.score_changes),
            ]
            for item in history
        ],
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]
