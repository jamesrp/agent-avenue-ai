from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from agent_avenue.runners import (
    POPULATION_PAIR_COUNT,
    PopulationExperimentConfig,
    build_population_experiment_plan,
    build_population_policy_bundle,
    nested_population_bootstrap,
    population_arena_cells,
    population_decision,
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
    assert (
        "frozen_arena_total_conflict_7400_comparisons_plus_600_reference_games"
        in execution["claim_ineligibility_reasons"]
    )
    assert len(corpus["replicates"]) == 3
    assert all(
        len(replicate["assignments"][arm]["matchups"]) == POPULATION_PAIR_COUNT
        for replicate in corpus["replicates"]
        for arm in ("control", "treatment")
    )
    # Candidate comparisons preserve the agreement's 7,400 arithmetic; the separately required
    # parent reference is deliberately visible rather than silently omitted.
    assert arenas["comparison_games_per_replicate"] == 7_400
    assert arenas["required_parent_heuristic_reference_games_per_replicate"] == 600
    assert arenas["physical_games_per_replicate"] == 8_000


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
    direct = next(cell for cell in first_replicate if cell.key == "treatment-vs-control")

    assert treatment.master_seed == control.master_seed
    assert direct.master_seed != treatment.master_seed
    assert plan["corpus_plan_fingerprint"] == corpus_plan.fingerprint


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
