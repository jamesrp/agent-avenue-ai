from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agent_avenue.agents import RandomAgent
from agent_avenue.runners import (
    POPULATION_PAIR_COUNT,
    AgentSpec,
    ArenaConfig,
    PopulationExperimentConfig,
    build_population_experiment_plan,
    build_population_policy_bundle,
    nested_population_bootstrap,
    population_arena_cells,
    population_decision,
    schedule_arena,
)


def _toy_config(tmp_path: Path) -> PopulationExperimentConfig:
    paths = {
        policy_id: tmp_path / "inputs" / policy_id
        for policy_id in ("q0", "q1", "q2", "q3", "q4", "historical-q0")
    }
    return PopulationExperimentConfig(
        tmp_path / "output",
        checkpoint_paths=paths,
        holdout_roots=(tmp_path / "holdout",),
        smoke_pair_count=2,
        smoke_max_epochs=2,
    )


def _full_toy_config(tmp_path: Path) -> PopulationExperimentConfig:
    paths = {
        policy_id: tmp_path / "inputs" / policy_id
        for policy_id in ("q0", "q1", "q2", "q3", "q4", "historical-q0")
    }
    return PopulationExperimentConfig(
        tmp_path / "output",
        checkpoint_paths=paths,
        holdout_roots=(tmp_path / "holdout",),
    )


def test_population_experiment_plan_freezes_six_corpora_and_arena_schedule(
    tmp_path: Path,
) -> None:
    config = _full_toy_config(tmp_path)
    bundle = build_population_policy_bundle(config.checkpoint_paths, toy=True)
    first = build_population_experiment_plan(config, bundle)
    second = build_population_experiment_plan(config, bundle)

    assert first == second
    execution = first["execution"]
    corpus = first["corpus_plan"]
    arenas = first["arenas"]
    assert isinstance(execution, dict)
    assert isinstance(corpus, dict)
    assert isinstance(arenas, dict)
    assert execution["claim_eligible"] is False
    assert len(corpus["replicates"]) == 3
    assert all(
        len(replicate["assignments"][arm]["matchups"]) == POPULATION_PAIR_COUNT
        for replicate in corpus["replicates"]
        for arm in ("control", "treatment")
    )
    # Candidate comparisons preserve the agreement's 7,400 arithmetic; the separately required
    # parent reference is deliberately visible rather than silently omitted.
    assert arenas["candidate_comparison_games_per_replicate"] == 7_400
    assert arenas["parent_heuristic_reference_games_per_replicate"] == 600
    assert arenas["physical_games_per_replicate"] == 8_000
    assert arenas["total_physical_games"] == 24_000


def test_population_arena_shared_cells_have_identical_named_seed(tmp_path: Path) -> None:
    config = _toy_config(tmp_path)
    bundle = build_population_policy_bundle(config.checkpoint_paths, toy=True)
    plan = build_population_experiment_plan(config, bundle)
    from agent_avenue.runners.population_experiment import _corpus_plan_from_bundle

    corpus_plan = _corpus_plan_from_bundle(bundle)
    cells = population_arena_cells(corpus_plan, pair_count=config.pairs_per_cell)
    first_replicate = [cell for cell in cells if cell.replicate_id == "replicate-1"]
    treatment = next(cell for cell in first_replicate if cell.key == "treatment-vs-heuristic")
    control = next(cell for cell in first_replicate if cell.key == "control-vs-heuristic")
    reference = next(
        cell for cell in first_replicate if cell.key == "parent-vs-heuristic-reference"
    )
    direct = next(cell for cell in first_replicate if cell.key == "treatment-vs-control")

    assert treatment.master_seed == control.master_seed == reference.master_seed
    assert treatment.shared_group == control.shared_group == reference.shared_group == "heuristic"
    assert direct.master_seed != treatment.master_seed
    assert plan["corpus_plan_fingerprint"] == corpus_plan.fingerprint

    candidate_lane = "m7-population-replay-v1:replicate-1:candidate-lane"
    heuristic_lane = "replicate-1:heuristic"
    parent_lane = "replicate-1:q0"

    def spec(agent_id: str, lane: str) -> AgentSpec:
        return AgentSpec(agent_id, {"type": agent_id}, RandomAgent, rng_identity=lane)

    schedules = tuple(
        tuple(
            schedule_arena(ArenaConfig(cell.run_id, left, right, cell.pair_count, cell.master_seed))
        )
        for cell, left, right in (
            (
                treatment,
                spec("treatment", candidate_lane),
                spec("greedy-public-v1", heuristic_lane),
            ),
            (control, spec("control", candidate_lane), spec("greedy-public-v1", heuristic_lane)),
            (
                reference,
                spec("enveloped-q0", parent_lane),
                spec("greedy-public-v1", heuristic_lane),
            ),
        )
    )
    for treatment_game, control_game, parent_game in zip(*schedules, strict=True):
        assert treatment_game.game_id == control_game.game_id == parent_game.game_id
        assert treatment_game.setup_seed == control_game.setup_seed == parent_game.setup_seed
        heuristic_index = 1 if treatment_game.game_id.endswith("a-first") else 0
        assert all(
            game.seats[heuristic_index].rng_identity == heuristic_lane
            for game in (treatment_game, control_game, parent_game)
        )
        assert (
            len(
                {
                    game.agent_seeds[heuristic_index]
                    for game in (treatment_game, control_game, parent_game)
                }
            )
            == 1
        )
        candidate_index = 1 - heuristic_index
        assert treatment_game.seats[candidate_index].rng_identity == candidate_lane
        assert control_game.seats[candidate_index].rng_identity == candidate_lane
        assert (
            treatment_game.agent_seeds[candidate_index] == control_game.agent_seeds[candidate_index]
        )
        assert parent_game.seats[candidate_index].rng_identity == parent_lane


def test_corrected_default_arena_count_contract_rejects_tampering(tmp_path: Path) -> None:
    from agent_avenue.runners.population_experiment import _validate_corrected_default_arena_counts

    config = _full_toy_config(tmp_path)
    bundle = build_population_policy_bundle(config.checkpoint_paths, toy=True)
    plan = build_population_experiment_plan(config, bundle)
    _validate_corrected_default_arena_counts(plan)

    arenas = plan["arenas"]
    assert isinstance(arenas, dict)
    arenas["total_physical_games"] = 22_200
    with pytest.raises(ValueError, match="8,000-game"):
        _validate_corrected_default_arena_counts(plan)


def test_nested_bootstrap_and_decision_branches_are_deterministic() -> None:
    rows = ((1.0, 1.0), (1.0, 0.5), (1.0, 1.0))
    first = nested_population_bootstrap(rows, seed=17, domain="fixture")
    second = nested_population_bootstrap(rows, seed=17, domain="fixture")
    assert first == second
    assert first["resamples"] == 20_000

    nested = {
        "treatment_vs_control": {"interval": [0.51, 0.8]},
        "treatment_vs_parent": {"interval": [0.51, 0.8]},
        "treatment_vs_random": {"interval": [0.51, 0.8]},
        "treatment_minus_parent_heuristic": {"interval": [-0.01, 0.1]},
        "treatment_minus_control_greedy-public-v1": {"interval": [-0.01, 0.1]},
    }
    statistics = {
        "nested": nested,
        "replicates": [
            {"direct_treatment_vs_control": 0.6},
            {"direct_treatment_vs_control": 0.6},
            {"direct_treatment_vs_control": 0.4},
        ],
        "minimum_candidate_seat_rate": 0.45,
        "tactical_invariants_passed": True,
    }
    assert (
        population_decision(statistics, integrity_passed=True)["decision"]
        == "advancing_model_v1_collection_recipe"
    )
    nested["treatment_vs_control"] = {"interval": [0.49, 0.51]}
    assert (
        population_decision(statistics, integrity_passed=True)["decision"]
        == "inconclusive_does_not_advance"
    )


def test_default_learned_clean_plan_is_claim_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    from agent_avenue.learning import create_model, save_checkpoint
    from agent_avenue.runners import population_experiment
    from agent_avenue.storage.provenance import SourceIdentity

    paths: dict[str, Path] = {}
    for index, policy_id in enumerate(("q0", "q1", "q2", "q3", "q4", "historical-q0")):
        path = tmp_path / "learned-inputs" / policy_id
        save_checkpoint(
            path,
            create_model(800 + index),
            metrics={"fixture": True},
            training_config={"fixture": True},
            training_seeds={"fixture": index},
            metadata={"fixture": "population-plan"},
            created_at="2026-09-11T00:00:00+00:00",
        )
        paths[policy_id] = path

    monkeypatch.setattr(
        population_experiment,
        "inspect_source_identity",
        lambda: SourceIdentity("a" * 40, "b" * 64, True, "c" * 64),
    )
    config = PopulationExperimentConfig(
        tmp_path / "claim-output", checkpoint_paths=paths, holdout_roots=(tmp_path / "holdout",)
    )
    plan = build_population_experiment_plan(config, build_population_policy_bundle(paths))

    execution = plan["execution"]
    assert isinstance(execution, dict)
    assert execution["claim_eligible"] is True
    assert execution["claim_ineligibility_reasons"] == []


def test_population_experiment_import_path_stays_torch_free() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import agent_avenue.runners; "
            "assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""
