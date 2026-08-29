from dataclasses import replace
from pathlib import Path

import pytest

from agent_avenue.engine import (
    ErrorCode,
    Phase,
    ReplayError,
    create_replay,
    legal_actions,
    load_replay,
    new_game,
    replay,
    save_replay,
    state_fingerprint,
)
from agent_avenue.engine.replay import replay_from_data, replay_to_data


def _complete_game(seed: int = 55):
    state = new_game(seed=seed)
    for _ in range(200):
        actions = legal_actions(state)
        if not actions:
            break
        from agent_avenue.engine import apply_action

        state = apply_action(state, actions[-1])
    assert state.phase is Phase.TERMINAL
    return state


def test_replay_round_trip_without_hidden_snapshots(tmp_path: Path) -> None:
    final = _complete_game()
    record = create_replay(final.config, final.seed, final)
    path = tmp_path / "game.json"
    save_replay(record, path)
    loaded = load_replay(path)
    replayed = replay(loaded)
    assert replayed == final
    assert state_fingerprint(replayed) == record.final_fingerprint
    data = replay_to_data(record)
    assert "deck" not in data or data["deck"] is None
    assert "hands" not in data


def test_replay_rejects_versions_and_fingerprint_mismatch() -> None:
    final = _complete_game(56)
    record = create_replay(final.config, final.seed, final)
    with pytest.raises(ReplayError) as version_error:
        replay(replace(record, schema_version=999))
    assert version_error.value.code is ErrorCode.UNSUPPORTED_VERSION
    with pytest.raises(ReplayError) as fingerprint_error:
        replay(replace(record, final_fingerprint="0" * 64))
    assert fingerprint_error.value.code is ErrorCode.FINGERPRINT_MISMATCH


def test_malformed_replay_is_rejected() -> None:
    with pytest.raises(ReplayError) as error:
        replay_from_data({"schema_version": 1})
    assert error.value.code is ErrorCode.MALFORMED_REPLAY


def test_committed_replay_fixture_is_a_golden_compatibility_check() -> None:
    path = Path(__file__).parents[1] / "fixtures" / "scripted_seed17.replay.json"
    state = replay(load_replay(path))
    assert state_fingerprint(state) == (
        "869aa66a6345e600da94494d543c3d4db7c5db74e926206ca8deff61e57b0d6d"
    )


def test_replay_parser_rejects_type_coercion_and_unknown_fields() -> None:
    final = _complete_game(57)
    data = replay_to_data(create_replay(final.config, final.seed, final))
    bad_schema = dict(data)
    bad_schema["schema_version"] = True
    with pytest.raises(ReplayError):
        replay_from_data(bad_schema)
    extra = dict(data)
    extra["unexpected"] = "field"
    with pytest.raises(ReplayError):
        replay_from_data(extra)


def test_replay_rejects_expected_outcome_mismatch() -> None:
    final = _complete_game(58)
    record = create_replay(final.config, final.seed, final)
    assert record.expected_outcome is not None
    wrong_outcome = replace(
        record.expected_outcome,
        winner=record.expected_outcome.winner.other(),
    )
    with pytest.raises(ReplayError) as error:
        replay(replace(record, expected_outcome=wrong_outcome))
    assert error.value.code is ErrorCode.OUTCOME_MISMATCH
