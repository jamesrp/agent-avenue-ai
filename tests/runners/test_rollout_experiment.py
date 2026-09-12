from __future__ import annotations

import ast
from pathlib import Path

import pytest

from agent_avenue.engine import new_game
from agent_avenue.observation import observe
from agent_avenue.rollout import safe_position_identity
from agent_avenue.runners.rollout_experiment import (
    RolloutExperimentConfig,
    RolloutExperimentError,
    _cells,
    _check_cardinality,
)


def test_claim_and_smoke_schedule_cardinalities_are_explicit(tmp_path: Path) -> None:
    claim = RolloutExperimentConfig(
        output=tmp_path / "claim",
        step3_root=tmp_path / "runs/m7-structured-model-v2",
        holdout_roots=(tmp_path / "runs",),
        claim=True,
    )
    claim_cells = _cells(claim)
    _check_cardinality(claim_cells, claim=True)
    assert len(claim_cells) == 54
    assert sum(cell.pair_count * 2 for cell in claim_cells) == 24_000

    smoke = RolloutExperimentConfig(
        output=tmp_path / "smoke",
        step3_root=tmp_path / "runs/m7-structured-model-v2",
        holdout_roots=(tmp_path / "runs",),
        smoke_positions_per_stratum=1,
        smoke_treatment_epochs=1,
        smoke_arena_pairs=1,
    )
    smoke_cells = _cells(smoke)
    _check_cardinality(smoke_cells, claim=False)
    assert [(cell.key, cell.pair_count) for cell in smoke_cells] == [("T1-vs-C1", 1)]


def test_smoke_rejects_hidden_claim_like_defaults(tmp_path: Path) -> None:
    with pytest.raises(RolloutExperimentError, match="only supported smoke"):
        RolloutExperimentConfig(
            output=tmp_path / "smoke",
            step3_root=tmp_path / "runs/m7-structured-model-v2",
            holdout_roots=(tmp_path / "runs",),
            smoke_positions_per_stratum=2,
            smoke_treatment_epochs=1,
            smoke_arena_pairs=1,
        )


def test_safe_identity_excludes_replicate_and_stratum_metadata() -> None:
    state = new_game(seed=17)
    observation = observe(state, state.active_player)
    assert safe_position_identity(observation) == safe_position_identity(observation)


def test_independent_validator_has_no_forbidden_rollout_or_aggregate_imports() -> None:
    path = Path("scripts/validate_counterfactual_rollout_supervision_v1.py")
    tree = ast.parse(path.read_text())
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    forbidden = {
        "agent_avenue.rollout.identity",
        "agent_avenue.rollout.latent",
        "agent_avenue.rollout.targets",
        "agent_avenue.rollout.artifact",
        "agent_avenue.learning.rollout_train",
        "agent_avenue.runners.rollout_experiment",
        "agent_avenue.runners.arena",
    }
    assert not imports.intersection(forbidden)
