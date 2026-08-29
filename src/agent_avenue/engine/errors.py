"""Structured engine and replay errors."""

from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_CONFIG = "invalid_config"
    INVALID_STATE = "invalid_state"
    WRONG_ACTION_TYPE = "wrong_action_type"
    WRONG_ACTOR = "wrong_actor"
    ILLEGAL_ACTION = "illegal_action"
    STALE_ACTION = "stale_action"
    GAME_FINISHED = "game_finished"
    UNSUPPORTED_VERSION = "unsupported_version"
    FINGERPRINT_MISMATCH = "fingerprint_mismatch"
    OUTCOME_MISMATCH = "outcome_mismatch"
    MALFORMED_REPLAY = "malformed_replay"


class EngineError(Exception):
    """An error with stable machine-readable code and context."""

    def __init__(self, code: ErrorCode, message: str, **context: object) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


class ReplayError(EngineError):
    """Replay validation or verification failure."""
