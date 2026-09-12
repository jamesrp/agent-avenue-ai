from __future__ import annotations

import hashlib

import pytest

from agent_avenue.engine import PlayerId, new_game
from agent_avenue.observation import observe
from agent_avenue.rollout import (
    PanelPosition,
    RolloutSample,
    RolloutTargetRow,
    RolloutTargetSet,
    canonical_observation,
    safe_position_identity,
    stratum_for_observation,
)
from agent_avenue.rollout.artifact import (
    RolloutArtifactError,
    load_rollout_targets,
    save_rollout_targets,
)


def _targets() -> RolloutTargetSet:
    state = new_game(seed=23)
    observation = canonical_observation(observe(state, PlayerId.PLAYER_ONE))
    stratum = stratum_for_observation(observation)
    assert stratum is not None
    safe_id = safe_position_identity(observation)
    rank = hashlib.sha256(f"step4:panel:replicate-1:{stratum}:{safe_id}".encode()).hexdigest()
    position = PanelPosition("replicate-1", stratum, safe_id, rank, observation, "0" * 64)
    samples = tuple(
        RolloutSample(
            index,
            index + 1,
            "a" * 64,
            "q0",
            "random",
            (),
            True,
            False,
            1,
            0.5,
        )
        for index in range(10)
    )
    rows = tuple(
        RolloutTargetRow(
            "replicate-1",
            stratum,
            safe_id,
            action,
            (0.0,) * 519,
            0.5,
            samples,
        )
        for action in observation.legal_actions
    )
    return RolloutTargetSet((position,), rows)


def test_rollout_target_npz_manifest_round_trip_and_tamper_rejection(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path, manifest_path = save_rollout_targets(
        _targets(), tmp_path / "targets.npz", input_identities={"inputs": "frozen"}
    )
    loaded = load_rollout_targets(path)
    assert loaded.features.shape[1] == 519
    assert loaded.targets.shape[0] == len(_targets().rows)

    manifest_path.write_text(manifest_path.read_text().replace("counterfactual", "forged", 1))
    with pytest.raises(RolloutArtifactError, match="fingerprint"):
        load_rollout_targets(path)
