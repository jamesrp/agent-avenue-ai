"""Versioned semantic replay records and deterministic verification."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cards import CardName
from .errors import EngineError, ErrorCode, ReplayError
from .model import (
    Action,
    GameConfig,
    GameState,
    OfferSlot,
    OutcomeReason,
    OutcomeResolution,
    PlayerId,
    PlayOfferAction,
    RecruitAction,
    TerminalFacts,
    TerminalOutcome,
)
from .setup import (
    RULES_VERSION,
    SHUFFLE_VERSION,
    config_from_normalized,
    new_game,
    normalize_config,
)
from .transitions import apply_action, validate_state

REPLAY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    """Everything required to reconstruct a game without hidden snapshots."""

    schema_version: int
    rules_version: str
    shuffle_version: str
    config: GameConfig
    seed: int
    actions: tuple[Action, ...]
    final_fingerprint: str
    expected_outcome: TerminalOutcome | None


def _action_data(action: Action) -> dict[str, object]:
    if isinstance(action, PlayOfferAction):
        return {
            "type": "play_offer",
            "revision": action.revision,
            "actor": action.actor.value,
            "face_up": action.face_up.value,
            "face_down": action.face_down.value,
        }
    return {
        "type": "recruit",
        "revision": action.revision,
        "actor": action.actor.value,
        "slot": action.slot.value,
    }


def _outcome_data(outcome: TerminalOutcome | None) -> dict[str, object] | None:
    if outcome is None:
        return None
    facts = outcome.facts
    return {
        "winner": outcome.winner.value,
        "reason": outcome.reason.value,
        "resolution": outcome.resolution.value,
        "active_player": outcome.active_player.value,
        "turn": outcome.turn,
        "facts": {
            "scores": list(facts.scores),
            "codebreaker_counts": list(facts.codebreaker_counts),
            "daredevil_counts": list(facts.daredevil_counts),
            "score_gap_winners": [player.value for player in facts.score_gap_winners],
            "instant_winners": [player.value for player in facts.instant_winners],
            "instant_losers": [player.value for player in facts.instant_losers],
            "candidate_winners": [player.value for player in facts.candidate_winners],
            "deck_empty": facts.deck_empty,
            "next_player_hand_size": facts.next_player_hand_size,
        },
    }


def _state_data(state: GameState) -> dict[str, object]:
    """Canonical trusted serialization used only for deterministic fingerprints."""
    return {
        "config": normalize_config(state.config),
        "seed": state.seed,
        "deck": [card.value for card in state.deck],
        "hands": [[card.value for card in hand] for hand in state.hands],
        "recruited": [[card.value for card in cards] for cards in state.recruited],
        "scores": list(state.scores),
        "active_player": state.active_player.value,
        "turn": state.turn,
        "phase": state.phase.value,
        "revision": state.revision,
        "offer": None
        if state.offer is None
        else {
            "offered_by": state.offer.offered_by.value,
            "face_up": state.offer.face_up.value,
            "face_down": state.offer.face_down.value,
        },
        "actions": [_action_data(action) for action in state.actions],
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
            for item in state.history
        ],
        "outcome": _outcome_data(state.outcome),
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def state_fingerprint(state: GameState) -> str:
    """Return a stable SHA-256 fingerprint of complete authoritative state."""
    return hashlib.sha256(_canonical_json(_state_data(state)).encode()).hexdigest()


def create_replay(initial_config: GameConfig, seed: int, final_state: GameState) -> ReplayRecord:
    """Build and verify a replay record from a state and its semantic history."""
    validate_state(final_state)
    if initial_config != final_state.config or seed != final_state.seed:
        raise ReplayError(
            ErrorCode.MALFORMED_REPLAY,
            "replay provenance does not match final state",
        )
    record = ReplayRecord(
        schema_version=REPLAY_SCHEMA_VERSION,
        rules_version=RULES_VERSION,
        shuffle_version=SHUFFLE_VERSION,
        config=initial_config,
        seed=seed,
        actions=final_state.actions,
        final_fingerprint=state_fingerprint(final_state),
        expected_outcome=final_state.outcome,
    )
    replayed = replay(record)
    if replayed != final_state:
        raise ReplayError(ErrorCode.FINGERPRINT_MISMATCH, "state is not reproducible from actions")
    return record


def replay(record: ReplayRecord) -> GameState:
    """Reconstruct and verify a replay record."""
    if record.schema_version != REPLAY_SCHEMA_VERSION:
        raise ReplayError(
            ErrorCode.UNSUPPORTED_VERSION,
            "unsupported replay schema version",
            actual=record.schema_version,
            expected=REPLAY_SCHEMA_VERSION,
        )
    if record.rules_version != RULES_VERSION or record.shuffle_version != SHUFFLE_VERSION:
        raise ReplayError(
            ErrorCode.UNSUPPORTED_VERSION,
            "unsupported rules or shuffle version",
            rules_version=record.rules_version,
            shuffle_version=record.shuffle_version,
        )
    state = new_game(record.config, record.seed)
    for index, action in enumerate(record.actions):
        try:
            state = apply_action(state, action)
        except Exception as exc:
            raise ReplayError(
                ErrorCode.MALFORMED_REPLAY,
                "replay action could not be applied",
                action_index=index,
            ) from exc
    actual_fingerprint = state_fingerprint(state)
    if actual_fingerprint != record.final_fingerprint:
        raise ReplayError(
            ErrorCode.FINGERPRINT_MISMATCH,
            "replayed state fingerprint does not match",
            expected=record.final_fingerprint,
            actual=actual_fingerprint,
        )
    actual_outcome = state.outcome
    if actual_outcome != record.expected_outcome:
        raise ReplayError(
            ErrorCode.OUTCOME_MISMATCH,
            "replayed terminal outcome does not match",
            expected=_outcome_data(record.expected_outcome),
            actual=_outcome_data(actual_outcome),
        )
    return state


def replay_to_data(record: ReplayRecord) -> dict[str, object]:
    """Serialize a record to its normalized JSON-ready form."""
    return {
        "schema_version": record.schema_version,
        "rules_version": record.rules_version,
        "shuffle_version": record.shuffle_version,
        "config": normalize_config(record.config),
        "seed": record.seed,
        "actions": [_action_data(action) for action in record.actions],
        "final_fingerprint": record.final_fingerprint,
        "expected_outcome": _outcome_data(record.expected_outcome),
    }


def _parse_action(value: object) -> Action:
    if not isinstance(value, dict):
        raise ValueError
    kind = value.get("type")
    if kind == "play_offer":
        if set(value) != {"type", "revision", "actor", "face_up", "face_down"}:
            raise ValueError
    elif kind == "recruit":
        if set(value) != {"type", "revision", "actor", "slot"}:
            raise ValueError
    else:
        raise ValueError
    revision = _exact_int(value["revision"])
    actor = value["actor"]
    if not isinstance(actor, str):
        raise ValueError
    if kind == "play_offer":
        return PlayOfferAction(
            revision,
            PlayerId(actor),
            CardName(value["face_up"]),
            CardName(value["face_down"]),
        )
    return RecruitAction(revision, PlayerId(actor), OfferSlot(value["slot"]))


def _exact_int(value: object) -> int:
    if type(value) is not int:
        raise ValueError
    return value


def _parse_player_list(value: object) -> tuple[PlayerId, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError
    return tuple(PlayerId(item) for item in value)


def _parse_outcome(value: object) -> TerminalOutcome | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "winner",
        "reason",
        "resolution",
        "active_player",
        "turn",
        "facts",
    }:
        raise ValueError
    facts_value = value["facts"]
    if not isinstance(facts_value, dict) or set(facts_value) != {
        "scores",
        "codebreaker_counts",
        "daredevil_counts",
        "score_gap_winners",
        "instant_winners",
        "instant_losers",
        "candidate_winners",
        "deck_empty",
        "next_player_hand_size",
    }:
        raise ValueError

    def pair(name: str) -> tuple[int, int]:
        raw = facts_value[name]
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError
        return (_exact_int(raw[0]), _exact_int(raw[1]))

    deck_empty = facts_value["deck_empty"]
    if type(deck_empty) is not bool:
        raise ValueError

    facts = TerminalFacts(
        scores=pair("scores"),
        codebreaker_counts=pair("codebreaker_counts"),
        daredevil_counts=pair("daredevil_counts"),
        score_gap_winners=_parse_player_list(facts_value["score_gap_winners"]),
        instant_winners=_parse_player_list(facts_value["instant_winners"]),
        instant_losers=_parse_player_list(facts_value["instant_losers"]),
        candidate_winners=_parse_player_list(facts_value["candidate_winners"]),
        deck_empty=deck_empty,
        next_player_hand_size=_exact_int(facts_value["next_player_hand_size"]),
    )
    return TerminalOutcome(
        winner=PlayerId(value["winner"]),
        reason=OutcomeReason(value["reason"]),
        resolution=OutcomeResolution(value["resolution"]),
        active_player=PlayerId(value["active_player"]),
        turn=_exact_int(value["turn"]),
        facts=facts,
    )


def replay_from_data(value: object) -> ReplayRecord:
    """Parse a normalized replay object with exact fields and JSON types."""
    required = {
        "schema_version",
        "rules_version",
        "shuffle_version",
        "config",
        "seed",
        "actions",
        "final_fingerprint",
        "expected_outcome",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ReplayError(ErrorCode.MALFORMED_REPLAY, "replay has invalid fields")
    try:
        actions_value = value["actions"]
        if not isinstance(actions_value, list):
            raise ValueError
        rules_version = value["rules_version"]
        shuffle_version = value["shuffle_version"]
        fingerprint = value["final_fingerprint"]
        if not all(isinstance(item, str) for item in (rules_version, shuffle_version, fingerprint)):
            raise ValueError
        if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
            raise ValueError
        record = ReplayRecord(
            schema_version=_exact_int(value["schema_version"]),
            rules_version=rules_version,
            shuffle_version=shuffle_version,
            config=config_from_normalized(value["config"]),
            seed=_exact_int(value["seed"]),
            actions=tuple(_parse_action(item) for item in actions_value),
            final_fingerprint=fingerprint,
            expected_outcome=_parse_outcome(value["expected_outcome"]),
        )
    except (KeyError, TypeError, ValueError, EngineError) as exc:
        raise ReplayError(ErrorCode.MALFORMED_REPLAY, "malformed replay record") from exc
    return record


def save_replay(record: ReplayRecord, path: Path) -> None:
    """Write canonical replay JSON."""
    path.write_text(_canonical_json(replay_to_data(record)) + "\n")


def load_replay(path: Path) -> ReplayRecord:
    """Load replay JSON from disk."""
    try:
        value: Any = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(ErrorCode.MALFORMED_REPLAY, "unable to read replay JSON") from exc
    return replay_from_data(value)
