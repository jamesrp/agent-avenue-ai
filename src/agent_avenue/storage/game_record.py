"""Compact, versioned records for completed automated games."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agent_avenue.agents import RNG_ALGORITHM
from agent_avenue.engine import GameState, PlayerId, ReplayRecord, create_replay, replay
from agent_avenue.engine.replay import replay_from_data, replay_to_data
from agent_avenue.engine.setup import RULES_VERSION, SHUFFLE_VERSION, normalize_config

from .fingerprints import code_fingerprint, rules_fingerprint

GAME_RECORD_SCHEMA_VERSION = 1


class GameRecordError(ValueError):
    """Raised when a completed-game record is malformed or incompatible."""


@dataclass(frozen=True, slots=True)
class AgentSeatRecord:
    player: PlayerId
    agent_id: str
    config: Mapping[str, object]
    seed: int
    rng_algorithm: str = RNG_ALGORITHM
    seed_derivation: str = "supplied"
    rng_domain: str = ""

    @property
    def rng_identity(self) -> str:
        """Return the logical identity encoded by the agent RNG domain."""
        prefix = "agent:"
        return self.rng_domain[len(prefix) :] if self.rng_domain.startswith(prefix) else ""


@dataclass(frozen=True, slots=True)
class GameRecord:
    schema_version: int
    run_id: str
    game_id: str
    pair_id: str | None
    created_at: str
    rules_version: str
    shuffle_version: str
    rules_fingerprint: str
    code_fingerprint: str
    seats: tuple[AgentSeatRecord, AgentSeatRecord]
    replay: ReplayRecord
    winner: PlayerId
    terminal_reason: str
    final_scores: tuple[int, int]
    turn_count: int
    decision_count: int


def _copy_config(config: Mapping[str, object]) -> dict[str, object]:
    value = json.loads(json.dumps(dict(config), sort_keys=True))
    if not isinstance(value, dict):  # pragma: no cover
        raise GameRecordError("agent configuration must be a JSON object")
    return value


def create_game_record(
    *,
    run_id: str,
    game_id: str,
    pair_id: str | None,
    seats: tuple[AgentSeatRecord, AgentSeatRecord],
    final_state: GameState,
    created_at: datetime | None = None,
) -> GameRecord:
    """Create a self-verifying record from a terminal state."""
    if final_state.outcome is None:
        raise GameRecordError("completed-game records require a terminal state")
    if tuple(seat.player for seat in seats) != tuple(PlayerId):
        raise GameRecordError("seat records must be ordered player one, player two")
    replay_record = create_replay(final_state.config, final_state.seed, final_state)
    outcome = final_state.outcome
    detached_seats = tuple(
        AgentSeatRecord(
            seat.player,
            seat.agent_id,
            MappingProxyType(_copy_config(seat.config)),
            seat.seed,
            seat.rng_algorithm,
            seat.seed_derivation,
            seat.rng_domain,
        )
        for seat in seats
    )
    return GameRecord(
        schema_version=GAME_RECORD_SCHEMA_VERSION,
        run_id=run_id,
        game_id=game_id,
        pair_id=pair_id,
        created_at=(created_at or datetime.now(UTC)).isoformat(),
        rules_version=RULES_VERSION,
        shuffle_version=SHUFFLE_VERSION,
        rules_fingerprint=rules_fingerprint(),
        code_fingerprint=code_fingerprint(),
        seats=(detached_seats[0], detached_seats[1]),
        replay=replay_record,
        winner=outcome.winner,
        terminal_reason=outcome.reason.value,
        final_scores=final_state.scores,
        turn_count=len(final_state.history),
        decision_count=len(final_state.actions),
    )


def verify_game_record(record: GameRecord, *, verify_code: bool = True) -> GameState:
    """Replay a record and reject metadata, outcome, or fingerprint mismatches."""
    if tuple(seat.player for seat in record.seats) != tuple(PlayerId):
        raise GameRecordError("seat records must be ordered player one, player two")
    if any(
        type(seat.seed) is not int
        or not seat.agent_id
        or seat.rng_algorithm != RNG_ALGORITHM
        or not seat.seed_derivation
        or not seat.rng_identity
        for seat in record.seats
    ):
        raise GameRecordError("seat records contain invalid identity or RNG metadata")
    if record.schema_version != GAME_RECORD_SCHEMA_VERSION:
        raise GameRecordError("unsupported game-record schema version")
    if record.rules_version != RULES_VERSION or record.shuffle_version != SHUFFLE_VERSION:
        raise GameRecordError("unsupported rules or shuffle version")
    if record.rules_fingerprint != rules_fingerprint():
        raise GameRecordError("rules fingerprint mismatch")
    if verify_code and record.code_fingerprint != code_fingerprint():
        raise GameRecordError("code fingerprint mismatch")
    state = replay(record.replay)
    if state.outcome is None:
        raise GameRecordError("record replay is not terminal")
    expected = (
        state.outcome.winner,
        state.outcome.reason.value,
        state.scores,
        len(state.history),
        len(state.actions),
    )
    actual = (
        record.winner,
        record.terminal_reason,
        record.final_scores,
        record.turn_count,
        record.decision_count,
    )
    if actual != expected:
        raise GameRecordError("derived game metadata does not match replay")
    return state


def _seat_data(seat: AgentSeatRecord) -> dict[str, object]:
    return {
        "player": seat.player.value,
        "agent_id": seat.agent_id,
        "config": _copy_config(seat.config),
        "seed": seat.seed,
        "rng_algorithm": seat.rng_algorithm,
        "seed_derivation": seat.seed_derivation,
        "rng_domain": seat.rng_domain,
    }


def _game_record_data(record: GameRecord) -> dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "run_id": record.run_id,
        "game_id": record.game_id,
        "pair_id": record.pair_id,
        "created_at": record.created_at,
        "rules_version": record.rules_version,
        "shuffle_version": record.shuffle_version,
        "rules_fingerprint": record.rules_fingerprint,
        "code_fingerprint": record.code_fingerprint,
        "engine_config": normalize_config(record.replay.config),
        "setup_seed": record.replay.seed,
        "seats": [_seat_data(seat) for seat in record.seats],
        "replay": replay_to_data(record.replay),
        "winner": record.winner.value,
        "terminal_reason": record.terminal_reason,
        "final_scores": list(record.final_scores),
        "turn_count": record.turn_count,
        "decision_count": record.decision_count,
    }


def game_record_fingerprint(record: GameRecord) -> str:
    """Fingerprint deterministic game semantics, excluding creation time."""
    data = _game_record_data(record)
    data.pop("created_at")
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def game_record_to_data(record: GameRecord) -> dict[str, object]:
    data = _game_record_data(record)
    data["record_fingerprint"] = game_record_fingerprint(record)
    return data


def _exact_int(value: object) -> int:
    if type(value) is not int:
        raise GameRecordError("expected integer")
    return value


def game_record_from_data(value: object) -> GameRecord:
    required = {
        "schema_version",
        "run_id",
        "game_id",
        "pair_id",
        "created_at",
        "rules_version",
        "shuffle_version",
        "rules_fingerprint",
        "code_fingerprint",
        "engine_config",
        "setup_seed",
        "seats",
        "replay",
        "winner",
        "terminal_reason",
        "final_scores",
        "turn_count",
        "decision_count",
        "record_fingerprint",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise GameRecordError("game record has invalid fields")
    try:
        string_fields = (
            "run_id",
            "game_id",
            "created_at",
            "rules_version",
            "shuffle_version",
            "rules_fingerprint",
            "code_fingerprint",
            "terminal_reason",
        )
        if any(not isinstance(value[name], str) for name in string_fields):
            raise GameRecordError("game record contains non-string metadata")
        pair_id = value["pair_id"]
        if pair_id is not None and not isinstance(pair_id, str):
            raise GameRecordError("pair_id must be a string or null")
        raw_seats = value["seats"]
        if not isinstance(raw_seats, list) or len(raw_seats) != 2:
            raise GameRecordError("game record requires two seats")
        seats: list[AgentSeatRecord] = []
        for raw in raw_seats:
            if not isinstance(raw, dict) or set(raw) != {
                "player",
                "agent_id",
                "config",
                "seed",
                "rng_algorithm",
                "seed_derivation",
                "rng_domain",
            }:
                raise GameRecordError("malformed seat record")
            if (
                not isinstance(raw["agent_id"], str)
                or not isinstance(raw["config"], dict)
                or not isinstance(raw["rng_algorithm"], str)
                or not isinstance(raw["seed_derivation"], str)
                or not isinstance(raw["rng_domain"], str)
            ):
                raise GameRecordError("malformed seat configuration")
            seats.append(
                AgentSeatRecord(
                    PlayerId(raw["player"]),
                    raw["agent_id"],
                    MappingProxyType(_copy_config(raw["config"])),
                    _exact_int(raw["seed"]),
                    raw["rng_algorithm"],
                    raw["seed_derivation"],
                    raw["rng_domain"],
                )
            )
        if tuple(seat.player for seat in seats) != tuple(PlayerId):
            raise GameRecordError("seat records must be ordered player one, player two")
        scores = value["final_scores"]
        if not isinstance(scores, list) or len(scores) != 2:
            raise GameRecordError("final_scores must contain two values")
        record = GameRecord(
            schema_version=_exact_int(value["schema_version"]),
            run_id=value["run_id"],
            game_id=value["game_id"],
            pair_id=pair_id,
            created_at=value["created_at"],
            rules_version=value["rules_version"],
            shuffle_version=value["shuffle_version"],
            rules_fingerprint=value["rules_fingerprint"],
            code_fingerprint=value["code_fingerprint"],
            seats=(seats[0], seats[1]),
            replay=replay_from_data(value["replay"]),
            winner=PlayerId(value["winner"]),
            terminal_reason=value["terminal_reason"],
            final_scores=(_exact_int(scores[0]), _exact_int(scores[1])),
            turn_count=_exact_int(value["turn_count"]),
            decision_count=_exact_int(value["decision_count"]),
        )
        if not isinstance(value["record_fingerprint"], str):
            raise GameRecordError("record fingerprint must be a string")
        if normalize_config(record.replay.config) != value[
            "engine_config"
        ] or record.replay.seed != _exact_int(value["setup_seed"]):
            raise GameRecordError("duplicated setup metadata does not match replay")
        if value["record_fingerprint"] != game_record_fingerprint(record):
            raise GameRecordError("game record fingerprint mismatch")
        return record
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, GameRecordError):
            raise
        raise GameRecordError("malformed game record") from exc


def save_game_record(record: GameRecord, path: Path) -> None:
    path.write_text(
        json.dumps(game_record_to_data(record), sort_keys=True, separators=(",", ":")) + "\n"
    )


def load_game_record(path: Path) -> GameRecord:
    try:
        value: Any = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GameRecordError("unable to read game record") from exc
    return game_record_from_data(value)
