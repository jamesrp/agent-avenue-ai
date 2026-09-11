from __future__ import annotations

import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from agent_avenue.agents import (
    GreedyHeuristicConfig,
    RandomAgent,
    RandomAgentConfig,
)
from agent_avenue.runners import (
    POPULATION_EPSILON,
    POPULATION_PAIR_COUNT,
    POPULATION_POLICY_IDS,
    TREATMENT_LOGICAL_SLOT_COUNTS,
    AgentSpec,
    PolicySlotCount,
    PopulationCorpusConfig,
    PopulationMember,
    PopulationScheduleError,
    audit_population_alignment,
    audit_population_arm,
    build_population_corpus_plan,
    compose_terminal_population_agent,
    compose_terminal_population_config,
    enumerate_population_training_setup_blocks,
    reconstruct_population_assignment,
    reconstruct_population_matchup_counts,
    reconstruct_population_per_seat_marginals,
    reconstruct_population_setup_blocks,
    run_resumable_corpus,
)
from agent_avenue.runners.corpus import validate_record_matches_spec
from agent_avenue.storage import load_corpus


def _base_agent(policy_id: str) -> AgentSpec:
    config = RandomAgentConfig()
    return AgentSpec(
        f"raw-{policy_id}",
        config.to_data(),
        lambda config=config: RandomAgent(config),
    )


def _config() -> PopulationCorpusConfig:
    return PopulationCorpusConfig(
        members=tuple(
            PopulationMember(policy_id, _base_agent(policy_id))
            for policy_id in POPULATION_POLICY_IDS
        ),
        run_id_prefix="population-schedule-test",
    )


def _plan():  # type: ignore[no-untyped-def]
    return build_population_corpus_plan(_config())


def test_treatment_assignment_and_per_seat_marginals_are_exact() -> None:
    plan = _plan()
    replicate = plan.replicates[0]
    treatment = replicate.treatment_assignment

    assert len(treatment.matchups) == POPULATION_PAIR_COUNT
    assert len(treatment.logical_slots) == 4_000
    assert Counter(treatment.logical_slots) == {
        item.policy_id: item.logical_slots for item in TREATMENT_LOGICAL_SLOT_COUNTS
    }

    games = replicate.game_specs("treatment")
    audit = audit_population_arm(plan, replicate.replicate_id, "treatment", games)
    expected = tuple(TREATMENT_LOGICAL_SLOT_COUNTS)
    assert audit.player_one_marginals == expected
    assert audit.player_two_marginals == expected
    assert (
        reconstruct_population_assignment(plan, replicate.replicate_id, "treatment", games)
        == treatment
    )
    assert reconstruct_population_per_seat_marginals(
        plan, replicate.replicate_id, "treatment", games
    ) == (expected, expected)
    assert reconstruct_population_setup_blocks(
        plan, replicate.replicate_id, "treatment", games
    ) == (audit.setup_blocks)
    assert (
        sum(
            item.paired_blocks
            for item in reconstruct_population_matchup_counts(
                plan, replicate.replicate_id, "treatment", games
            )
        )
        == POPULATION_PAIR_COUNT
    )


def test_assignment_fingerprint_is_deterministic_and_complete() -> None:
    first = _plan()
    second = _plan()

    assert first.assignment_fingerprint == second.assignment_fingerprint
    assert first.fingerprint == second.fingerprint
    assert len(first.assignment_fingerprint) == 64
    data = first.to_data()
    assert data["configuration"]["root_seed"] == 2026091102
    assert len(data["replicates"]) == 3
    assert len(data["replicates"][0]["seeds"]["pair_seeds"]) == POPULATION_PAIR_COUNT
    assert (
        len(data["replicates"][0]["assignments"]["treatment"]["matchups"]) == POPULATION_PAIR_COUNT
    )


def test_control_and_treatment_have_common_setup_and_lane_rng_alignment() -> None:
    plan = _plan()
    replicate = plan.replicates[1]
    control = replicate.game_specs("control")
    treatment = replicate.game_specs("treatment")
    alignment = audit_population_alignment(control, treatment)

    assert alignment.paired_block_count == POPULATION_PAIR_COUNT
    assert alignment.game_count_per_arm == 4_000
    for control_game, treatment_game in zip(control, treatment, strict=True):
        assert control_game.game_id == treatment_game.game_id
        assert control_game.pair_id == treatment_game.pair_id
        assert control_game.setup_seed == treatment_game.setup_seed
        assert control_game.agent_seeds == treatment_game.agent_seeds
        assert control_game.agent_rng_domains == treatment_game.agent_rng_domains


def test_seat_swap_preserves_logical_lanes_independent_of_assigned_policy() -> None:
    replicate = _plan().replicates[0]
    first, second = replicate.game_specs("treatment")[:2]

    assert first.setup_seed == second.setup_seed
    assert first.pair_id == second.pair_id
    assert first.agent_seeds == tuple(reversed(second.agent_seeds))
    assert first.agent_rng_domains == tuple(reversed(second.agent_rng_domains))
    assert tuple(seat.agent_id for seat in first.seats) == tuple(
        reversed(tuple(seat.agent_id for seat in second.seats))
    )
    assert first.agent_seeds[0] != first.agent_seeds[1]
    assert first.agent_rng_domains[0] != first.agent_rng_domains[1]


def test_replicate_setup_families_and_training_holdout_candidates_are_disjoint() -> None:
    plan = _plan()
    setup_families = [
        {pair.setup_seed for pair in replicate.seed_plan.pair_seeds}
        for replicate in plan.replicates
    ]
    assert all(len(family) == POPULATION_PAIR_COUNT for family in setup_families)
    assert setup_families[0].isdisjoint(setup_families[1])
    assert setup_families[0].isdisjoint(setup_families[2])
    assert setup_families[1].isdisjoint(setup_families[2])

    candidates = enumerate_population_training_setup_blocks(plan)
    assert len(candidates) == 3 * POPULATION_PAIR_COUNT
    assert len({candidate.setup_identity for candidate in candidates}) == len(candidates)
    assert all(
        candidate.to_data()["corpus_arms"] == ["control", "treatment"] for candidate in candidates
    )
    with pytest.raises(TypeError):
        candidates[0].game_config["mutated"] = True


def test_control_assignment_is_q0_only_and_malformed_weights_fail_loudly() -> None:
    config = _config()
    control = _plan().replicates[0].control_assignment
    assert control.slot_counts == (PolicySlotCount("q0", 4_000),)
    assert set(control.logical_slots) == {"q0"}

    with pytest.raises(PopulationScheduleError, match="weights/counts"):
        PopulationCorpusConfig(
            members=config.members,
            run_id_prefix=config.run_id_prefix,
            treatment_slot_counts=(
                PolicySlotCount("q0", 1_599),
                *TREATMENT_LOGICAL_SLOT_COUNTS[1:],
            ),
        )
    with pytest.raises(PopulationScheduleError, match="2,000"):
        replace(config, pair_count=2)
    with pytest.raises(PopulationScheduleError, match="control assignment"):
        replace(
            _plan().replicates[0],
            control_assignment=_plan().replicates[0].treatment_assignment,
        )


def test_composition_preserves_raw_identity_and_normalizes_tactical_exploration_layers() -> None:
    base = _base_agent("q0")
    config = compose_terminal_population_config(base.config)
    spec = compose_terminal_population_agent(base, rng_identity="logical-lane-a")

    assert spec.agent_id == base.agent_id
    assert spec.rng_identity == "logical-lane-a"
    assert spec.config == config
    safety = config["base"]
    assert isinstance(safety, dict)
    epsilon = safety["base"]
    assert isinstance(epsilon, dict)
    raw = epsilon["base"]
    assert config["type"] == "terminal_offense"
    assert safety["type"] == "terminal_safety"
    assert epsilon["version"] == "epsilon-wrapper-v1"
    assert epsilon["epsilon"] == POPULATION_EPSILON.to_data()
    assert raw == base.config

    with pytest.raises(PopulationScheduleError, match="composed exactly once"):
        compose_terminal_population_config(config)


@pytest.mark.parametrize(
    ("base_config", "kind"),
    (
        (RandomAgentConfig().to_data(), "random"),
        (GreedyHeuristicConfig().to_data(), "heuristic"),
        (
            {
                "type": "learned_value",
                "version": "learned-value-agent-v1",
                "checkpoint_fingerprint": "a" * 64,
                "tensor_digest": "b" * 64,
            },
            "learned",
        ),
    ),
)
def test_composition_config_is_generic_across_population_base_kinds(
    base_config: dict[str, object], kind: str
) -> None:
    composed = compose_terminal_population_config(base_config)
    safety = composed["base"]
    assert isinstance(safety, dict)
    epsilon = safety["base"]
    assert isinstance(epsilon, dict)
    assert epsilon["base"] == base_config
    assert (
        epsilon["base"]["type"]
        == base_config["type"]
        == ("learned_value" if kind == "learned" else base_config["type"])
    )


def test_population_core_import_is_torch_free() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import agent_avenue.runners.population; "
            "assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""


def test_tiny_resumable_paired_corpora_validate_against_schedule_and_align_records(
    tmp_path: Path,
) -> None:
    plan = _plan()
    replicate_id = plan.replicates[0].replicate_id
    control_specs = plan.game_specs(replicate_id, "control")[:2]
    treatment_specs = plan.game_specs(replicate_id, "treatment")[:2]
    common_config = {"plan_fingerprint": plan.fingerprint, "fixture": "tiny-paired"}

    run_resumable_corpus(
        tmp_path / "control",
        control_specs,
        behavior_policy="population-control-fixture",
        root_seed=2026091102,
        generation=None,
        configuration={**common_config, "arm": "control"},
    )
    run_resumable_corpus(
        tmp_path / "treatment",
        treatment_specs,
        behavior_policy="population-treatment-fixture",
        root_seed=2026091102,
        generation=None,
        configuration={**common_config, "arm": "treatment"},
    )
    control_records = load_corpus(tmp_path / "control")[1]
    treatment_records = load_corpus(tmp_path / "treatment")[1]

    for record, spec in zip(control_records, control_specs, strict=True):
        validate_record_matches_spec(record, spec)
    for record, spec in zip(treatment_records, treatment_specs, strict=True):
        validate_record_matches_spec(record, spec)
    assert audit_population_alignment(control_records, treatment_records).paired_block_count == 1
