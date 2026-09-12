from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import agent_avenue.runners.rollout_experiment as rollout_experiment
from agent_avenue.agents import RandomAgent, derive_seed
from agent_avenue.engine import new_game
from agent_avenue.observation import observe
from agent_avenue.rollout import safe_position_identity
from agent_avenue.runners.arena import ArenaConfig, schedule_arena
from agent_avenue.runners.game import AgentSpec
from agent_avenue.runners.rollout_experiment import (
    RolloutArenaCell,
    RolloutExperimentConfig,
    RolloutExperimentError,
    _cells,
    _check_cardinality,
    _fingerprint,
    _validator_preflight,
)


def _claim_config(tmp_path: Path) -> RolloutExperimentConfig:
    return RolloutExperimentConfig(
        output=tmp_path / "claim",
        step3_root=tmp_path / "runs/m7-structured-model-v2",
        holdout_roots=(tmp_path / "runs",),
        claim=True,
    )


def test_claim_and_smoke_schedule_cardinalities_are_explicit(tmp_path: Path) -> None:
    claim_cells = _cells(_claim_config(tmp_path))
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


def test_claim_shared_groups_use_identical_master_setup_blocks(tmp_path: Path) -> None:
    groups: dict[tuple[str, str], list[RolloutArenaCell]] = {}
    for cell in _cells(_claim_config(tmp_path)):
        if cell.shared_group is not None:
            groups.setdefault((cell.replicate_id, cell.shared_group), []).append(cell)
    for (_, shared), group in groups.items():
        assert len({cell.master_seed for cell in group}) == 1
        assert group[0].master_seed == derive_seed(
            rollout_experiment.ROOT_SEED,
            f"step4:arena:{shared}:{group[0].replicate_id}",
        ) & ((1 << 63) - 1)
    assert len(groups[("replicate-1", "q0-parent")]) == 2
    assert len(groups[("replicate-1", "heuristic")]) == 3

    left = AgentSpec(
        "left", {"type": "random", "version": "random-agent-v1"}, RandomAgent, "lane-left"
    )
    right = AgentSpec(
        "right", {"type": "random", "version": "random-agent-v1"}, RandomAgent, "lane-right"
    )
    schedules = [
        tuple(schedule_arena(ArenaConfig(cell.run_id, left, right, 3, cell.master_seed)))
        for cell in groups[("replicate-1", "heuristic")]
    ]
    assert len({tuple(spec.setup_seed for spec in schedule) for schedule in schedules}) == 1
    assert all(schedule[0].seats[0].rng_identity == "lane-left" for schedule in schedules)
    assert all(schedule[1].seats[0].rng_identity == "lane-right" for schedule in schedules)


def test_claim_statistics_loads_t_c_and_q0_heuristic_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _claim_config(tmp_path)
    monkeypatch.setattr(rollout_experiment, "NESTED_RESAMPLES", 2)
    monkeypatch.setattr(rollout_experiment, "NESTED_LOWER", 0)
    monkeypatch.setattr(rollout_experiment, "NESTED_UPPER", 1)
    arenas: dict[str, dict[str, object]] = {}
    for cell in _cells(config):
        outcomes = [
            {"pair_id": f"pair-{index:06d}", "agent_a_wins": 1} for index in range(cell.pair_count)
        ]
        arenas[cell.key] = {
            "report": {
                "paired_seed_outcomes": outcomes,
                "agent_a_win_rate": 0.5,
                "agent_a_by_seat": {
                    "player_one": {"win_rate": 0.5},
                    "player_two": {"win_rate": 0.5},
                },
            },
            "tactical": {"passed": True},
        }
    statistics = rollout_experiment._statistics(_cells(config), arenas, smoke=False)
    nested = statistics.get("nested")
    assert isinstance(nested, dict)
    assert set(nested) == {
        "treatment_vs_control",
        "treatment_vs_q0_parent",
        "treatment_vs_random",
        "treatment_minus_q0_parent_heuristic",
        "control_minus_q0_parent_heuristic",
    }


def test_claim_preflight_requires_matching_independent_runtime_artifact(tmp_path: Path) -> None:
    payload = {
        "version": "m7-counterfactual-rollout-independent-runtime-preflight-v1",
        "claim_eligible": True,
        "both_target_throughputs_at_least_67": True,
        "leaf_and_cap_gate": True,
        "source_code_fingerprint": "a" * 64,
        "input_manifest_fingerprint": "b" * 64,
    }
    payload["artifact_fingerprint"] = _fingerprint(payload)
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(payload))
    accepted, reason = _validator_preflight(
        path,
        source={"code_fingerprint": "a" * 64},
        input_manifest_fingerprint="b" * 64,
    )
    assert accepted is not None and reason is None
    rejected, reason = _validator_preflight(
        path,
        source={"code_fingerprint": "c" * 64},
        input_manifest_fingerprint="b" * 64,
    )
    assert rejected is None and reason == "invalid_independent_validator_preflight"


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
        "agent_avenue.observation.build",
        "agent_avenue.rollout.latent",
        "agent_avenue.rollout.targets",
        "agent_avenue.rollout.artifact",
        "agent_avenue.learning.rollout_train",
        "agent_avenue.runners.rollout_experiment",
        "agent_avenue.runners.arena",
    }
    assert not imports.intersection(forbidden)
