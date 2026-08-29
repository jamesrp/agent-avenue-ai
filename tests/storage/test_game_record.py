from dataclasses import replace

import pytest

from agent_avenue.agents import RandomAgent, RandomAgentConfig
from agent_avenue.engine import GameConfig, PlayerId
from agent_avenue.runners import AgentSpec, GameSpec, run_game
from agent_avenue.storage import (
    GameRecordError,
    game_record_fingerprint,
    game_record_from_data,
    game_record_to_data,
    verify_game_record,
)


def _record():  # type: ignore[no-untyped-def]
    config = RandomAgentConfig()
    agent = AgentSpec("random-a", config.to_data(), RandomAgent)
    opponent = AgentSpec("random-b", config.to_data(), RandomAgent)
    return run_game(GameSpec("run", "game", "pair", GameConfig(), 17, (agent, opponent), (1, 2)))


def test_game_record_round_trip_replays_outcome_and_fingerprints() -> None:
    record = _record()
    loaded = game_record_from_data(game_record_to_data(record))
    final = verify_game_record(loaded)
    assert final.outcome is not None
    assert final.outcome.winner is record.winner
    assert final.scores == record.final_scores
    assert game_record_fingerprint(loaded) == game_record_fingerprint(record)
    assert tuple(seat.player for seat in record.seats) == tuple(PlayerId)


def test_record_verification_rejects_derived_metadata_tampering() -> None:
    record = _record()
    with pytest.raises(GameRecordError, match="metadata"):
        verify_game_record(replace(record, decision_count=record.decision_count + 1))


def test_record_parser_rejects_invalid_seat_order() -> None:
    data = game_record_to_data(_record())
    seats = data["seats"]
    assert isinstance(seats, list)
    assert isinstance(seats[0], dict)
    seats[0]["player"] = PlayerId.PLAYER_TWO.value
    with pytest.raises(GameRecordError, match="ordered"):
        game_record_from_data(data)


def test_record_detaches_agent_configurations() -> None:
    record = _record()
    data = game_record_to_data(record)
    seats = data["seats"]
    assert isinstance(seats, list)
    assert isinstance(seats[0], dict)
    config = seats[0]["config"]
    assert isinstance(config, dict)
    config["version"] = "mutated"
    assert game_record_to_data(record)["seats"] != seats


def test_record_parser_rejects_agent_metadata_tampering() -> None:
    data = game_record_to_data(_record())
    seats = data["seats"]
    assert isinstance(seats, list)
    assert isinstance(seats[0], dict)
    seats[0]["seed"] = 999
    with pytest.raises(GameRecordError, match="fingerprint"):
        game_record_from_data(data)


def test_record_parser_rejects_setup_metadata_tampering() -> None:
    data = game_record_to_data(_record())
    data["setup_seed"] = 999
    with pytest.raises(GameRecordError, match="setup metadata"):
        game_record_from_data(data)
