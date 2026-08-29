"""Deterministic two-player base-game engine public API."""

from .cards import CANONICAL_DECK, CARD_DEFINITIONS, CardName, recruit_effect
from .errors import EngineError, ErrorCode, ReplayError
from .model import (
    Action,
    Decision,
    GameConfig,
    GameState,
    OfferSlot,
    OutcomeReason,
    OutcomeResolution,
    Phase,
    PlayerId,
    PlayOfferAction,
    RecruitAction,
    TerminalOutcome,
)
from .replay import (
    ReplayRecord,
    create_replay,
    load_replay,
    replay,
    save_replay,
    state_fingerprint,
)
from .setup import new_game
from .transitions import apply_action, current_decision, legal_actions, validate_state

__all__ = [
    "CANONICAL_DECK",
    "CARD_DEFINITIONS",
    "Action",
    "CardName",
    "Decision",
    "EngineError",
    "ErrorCode",
    "GameConfig",
    "GameState",
    "OfferSlot",
    "OutcomeReason",
    "OutcomeResolution",
    "Phase",
    "PlayOfferAction",
    "PlayerId",
    "RecruitAction",
    "ReplayError",
    "ReplayRecord",
    "TerminalOutcome",
    "apply_action",
    "create_replay",
    "current_decision",
    "legal_actions",
    "load_replay",
    "new_game",
    "recruit_effect",
    "replay",
    "save_replay",
    "state_fingerprint",
    "validate_state",
]
