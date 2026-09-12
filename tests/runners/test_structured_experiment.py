from pathlib import Path

from agent_avenue.runners.structured_experiment import (
    ROOT_SEED,
    StructuredExperimentConfig,
    _cells,
    _joint_architecture_bootstrap,
    nested_bootstrap,
    structured_selection,
)


def _interval(point: float, lower: float, upper: float) -> dict[str, object]:
    return {"point_estimate": point, "interval": [lower, upper]}


def test_default_schedule_has_frozen_60_cells_and_30000_games() -> None:
    config = StructuredExperimentConfig(
        output=Path("/tmp/structured-output"),
        step2_root=Path("/tmp/runs/m7-population-replay-v1"),
    )
    cells = _cells(config)
    assert ROOT_SEED == 2026091203
    assert len(cells) == 60
    assert sum(cell.pair_count * 2 for cell in cells) == 30_000
    for replicate in ("replicate-1", "replicate-2", "replicate-3"):
        aligned = [
            cell
            for cell in cells
            if cell.replicate_id == replicate and cell.key.endswith("v2-vs-v1")
        ]
        assert len(aligned) == 2
        assert len({cell.master_seed for cell in aligned}) == 1
        assert {cell.shared_group for cell in aligned} == {"architecture"}


def test_smoke_schedule_only_truncates_declared_block_counts() -> None:
    config = StructuredExperimentConfig(
        output=Path("/tmp/structured-output"),
        step2_root=Path("/tmp/runs/m7-population-replay-v1"),
        smoke_pairs=1,
        smoke_max_epochs=1,
    )
    cells = _cells(config)
    assert len(cells) == 60
    assert {cell.pair_count for cell in cells} == {1}
    assert sum(cell.pair_count * 2 for cell in cells) == 120


def test_nested_bootstrap_and_joint_interaction_materialize_deterministically() -> None:
    rows = ((0.0, 1.0), (0.5, 0.5), (1.0, 0.0))
    first = nested_bootstrap(rows, domain="test")
    second = nested_bootstrap(rows, domain="test")
    assert first == second
    assert first["outer_draws_materialized_first"] is True
    control = ((0.0, 0.5),) * 3
    mixed = ((0.5, 1.0),) * 3
    interaction = _joint_architecture_bootstrap(control, mixed, interaction=True)
    assert interaction["point_estimate"] == 0.5
    assert interaction["blocks_per_replicate"] == 2


def test_selection_applies_candidate_only_seat_floor_and_development_label() -> None:
    nested = {
        "control_architecture_v2_minus_v1": _interval(0.55, 0.51, 0.59),
        "mixed_architecture_v2_minus_v1": _interval(0.51, 0.49, 0.55),
        "pooled_architecture_main_effect": _interval(0.53, 0.51, 0.56),
        "C_v2_vs_q0_parent": _interval(0.6, 0.55, 0.65),
        "M_v2_vs_q0_parent": _interval(0.6, 0.55, 0.65),
        "C_v2_vs_random": _interval(0.8, 0.7, 0.9),
        "M_v2_vs_random": _interval(0.8, 0.7, 0.9),
        "C_v2_minus_q0_parent_heuristic": _interval(0.0, -0.01, 0.02),
        "M_v2_minus_q0_parent_heuristic": _interval(0.0, -0.01, 0.02),
        "C_equal_eight_opponent_macro": _interval(0.7, 0.6, 0.8),
        "M_equal_eight_opponent_macro": _interval(0.69, 0.6, 0.8),
    }
    statistics = {
        "nested": nested,
        "minimum_v2_candidate_seat": {"C": 0.46, "M": 0.46},
        "tactical_invariants_passed": True,
        "per_replicate": [
            {
                "arms": {
                    "C": {"architecture_v2_minus_v1": 0.6},
                    "M": {"architecture_v2_minus_v1": 0.6},
                }
            },
            {
                "arms": {
                    "C": {"architecture_v2_minus_v1": 0.6},
                    "M": {"architecture_v2_minus_v1": 0.4},
                }
            },
            {
                "arms": {
                    "C": {"architecture_v2_minus_v1": 0.6},
                    "M": {"architecture_v2_minus_v1": 0.4},
                }
            },
        ],
    }
    result = structured_selection(statistics, integrity_passed=True)
    assert result["classification"] == "data_dependent_recipe_advance"
    assert result["development_selected_step4_input_recipe"] == "C"
    assert result["selection_label"] == "development-selected input; not advance/promotion"
