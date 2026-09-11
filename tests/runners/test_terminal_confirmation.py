from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agent_avenue.agents import (
    RandomAgent,
    RandomAgentConfig,
    ScriptedAgent,
    TerminalOffenseAgent,
    TerminalSafetyAgent,
)
from agent_avenue.engine import Phase, PlayerId, PlayOfferAction, apply_action, new_game
from agent_avenue.observation import observe
from agent_avenue.runners import (
    CONTROL_ID,
    DEFAULT_PAIRS_PER_FAMILY,
    FAMILY_IDS,
    FIELD_OPPONENTS,
    Q0_RNG_IDENTITY,
    TREATMENT_ID,
    AgentSpec,
    ArenaConfig,
    GameSpec,
    compare_matched_action_prefixes,
    confirmation_cells,
    family_master_seed,
    family_setup_seeds,
    independent_public_forced_win_oracle,
    run_game,
    schedule_arena,
    stratified_joint_bootstrap,
    validate_seed_families,
)


class FirstLegalAgent:
    def config_to_data(self) -> dict[str, object]:
        return {"type": "first-legal-test-agent", "version": "v1"}

    def choose_action(self, observation, decision, legal_actions, rng):  # type: ignore[no-untyped-def]
        del observation, decision, rng
        return legal_actions[0]


def _random_spec(agent_id: str, *, rng_identity: str | None = None) -> AgentSpec:
    config = RandomAgentConfig()
    return AgentSpec(agent_id, config.to_data(), RandomAgent, rng_identity=rng_identity)


def test_confirmation_seed_families_are_common_within_family_and_disjoint() -> None:
    cells = confirmation_cells()
    assert len(cells) == 30
    assert all(sum(cell.family == family for cell in cells) == 15 for family in FAMILY_IDS)
    assert {cell.opponent_id for cell in cells if cell.opponent_id is not None} == set(
        FIELD_OPPONENTS
    )

    validation = validate_seed_families(25)
    first = family_setup_seeds(FAMILY_IDS[0], 25)
    second = family_setup_seeds(FAMILY_IDS[1], 25)

    assert validation["status"] == "passed"
    assert len(set(first)) == len(first)
    assert len(set(second)) == len(second)
    assert set(first).isdisjoint(second)


def test_control_and_treatment_share_q0_rng_stream_against_shared_opponent() -> None:
    control = _random_spec(CONTROL_ID, rng_identity=Q0_RNG_IDENTITY)
    offense = TerminalOffenseAgent(TerminalSafetyAgent(RandomAgent()))
    treatment = AgentSpec(
        TREATMENT_ID,
        offense.config_to_data(),
        lambda: TerminalOffenseAgent(TerminalSafetyAgent(RandomAgent())),
        rng_identity=Q0_RNG_IDENTITY,
    )
    opponent = _random_spec("stable-opponent")
    master = family_master_seed(FAMILY_IDS[0])
    control_games = tuple(schedule_arena(ArenaConfig("control", control, opponent, 3, master)))
    treatment_games = tuple(
        schedule_arena(ArenaConfig("treatment", treatment, opponent, 3, master))
    )

    for control_game, treatment_game in zip(control_games, treatment_games, strict=True):
        assert control_game.setup_seed == treatment_game.setup_seed
        control_q0 = control_game.seats.index(control)
        treatment_q0 = treatment_game.seats.index(treatment)
        assert control_q0 == treatment_q0
        assert control_game.agent_seeds[control_q0] == treatment_game.agent_seeds[treatment_q0]
        opponent_index = 1 - control_q0
        assert (
            control_game.agent_seeds[opponent_index] == treatment_game.agent_seeds[opponent_index]
        )
        assert control_game.agent_rng_domains[control_q0] == "agent:q0-terminal-core-v1"


def test_agent_id_aliases_with_shared_rng_identity_preserve_behavior_and_outcome() -> None:
    first_agent = TerminalSafetyAgent(RandomAgent())
    second_agent = TerminalSafetyAgent(RandomAgent())
    first = AgentSpec(
        "behavior-alias-a",
        first_agent.config_to_data(),
        lambda: TerminalSafetyAgent(RandomAgent()),
        rng_identity="behavior-identical-core",
    )
    second = AgentSpec(
        "behavior-alias-b",
        second_agent.config_to_data(),
        lambda: TerminalSafetyAgent(RandomAgent()),
        rng_identity="behavior-identical-core",
    )
    opponent = _random_spec("alias-opponent")
    master_seed = 81173
    first_spec = next(schedule_arena(ArenaConfig("alias-a", first, opponent, 1, master_seed)))
    second_spec = next(schedule_arena(ArenaConfig("alias-b", second, opponent, 1, master_seed)))

    first_record = run_game(first_spec)
    second_record = run_game(second_spec)

    assert first_spec.setup_seed == second_spec.setup_seed
    assert first_spec.agent_seeds == second_spec.agent_seeds
    assert first_record.replay.actions == second_record.replay.actions
    assert first_record.replay.final_fingerprint == second_record.replay.final_fingerprint
    assert first_record.winner == second_record.winner
    assert first_record.final_scores == second_record.final_scores


def test_stratified_bootstrap_resamples_aligned_metrics_jointly() -> None:
    metrics = {
        FAMILY_IDS[0]: {
            "base": (0.0, 0.5, 1.0),
            "double": (0.0, 1.0, 2.0),
        },
        FAMILY_IDS[1]: {
            "base": (0.25, 0.75, 1.0),
            "double": (0.5, 1.5, 2.0),
        },
    }

    first = stratified_joint_bootstrap(metrics)
    second = stratified_joint_bootstrap(metrics)
    base = first["metrics"]["base"]
    double = first["metrics"]["double"]

    assert first == second
    assert double["point_estimate"] == 2 * base["point_estimate"]
    assert double["interval"] == [2 * value for value in base["interval"]]
    assert first["blocks_per_stratum"] == {family: 3 for family in FAMILY_IDS}


def test_matched_prefix_audit_accepts_alignment_until_guaranteed_win_endpoint() -> None:
    control_agent = TerminalSafetyAgent(FirstLegalAgent())
    treatment_agent = TerminalOffenseAgent(TerminalSafetyAgent(FirstLegalAgent()))
    control = AgentSpec(
        CONTROL_ID,
        control_agent.config_to_data(),
        lambda: TerminalSafetyAgent(FirstLegalAgent()),
        rng_identity=Q0_RNG_IDENTITY,
    )
    treatment = AgentSpec(
        TREATMENT_ID,
        treatment_agent.config_to_data(),
        lambda: TerminalOffenseAgent(TerminalSafetyAgent(FirstLegalAgent())),
        rng_identity=Q0_RNG_IDENTITY,
    )
    opponent = _random_spec("prefix-opponent")
    found = None
    for master_seed in range(1, 80):
        control_spec = next(
            schedule_arena(ArenaConfig("prefix-control", control, opponent, 1, master_seed))
        )
        treatment_spec = next(
            schedule_arena(ArenaConfig("prefix-treatment", treatment, opponent, 1, master_seed))
        )
        comparison = compare_matched_action_prefixes(
            run_game(control_spec), run_game(treatment_spec)
        )
        assert comparison["metadata_aligned"] is True
        assert comparison["prefix_aligned"] is True
        if comparison["guaranteed_win_endpoint_found"]:
            found = comparison
            break
    assert found is not None
    assert found["treatment_converted_at_endpoint"] is True
    assert found["endpoint_classification"] in {
        "control_missed_treatment_converted",
        "both_converted",
    }


def test_different_guaranteed_actions_are_valid_at_prefix_alignment_endpoint() -> None:
    control_agent = TerminalSafetyAgent(FirstLegalAgent())
    control = AgentSpec(
        CONTROL_ID,
        control_agent.config_to_data(),
        lambda: TerminalSafetyAgent(FirstLegalAgent()),
        rng_identity=Q0_RNG_IDENTITY,
    )
    opponent = _random_spec("prefix-regression-opponent")
    selected = None
    for master_seed in range(1, 500):
        control_spec = next(
            schedule_arena(ArenaConfig("prefix-regression", control, opponent, 1, master_seed))
        )
        record = run_game(control_spec)
        state = new_game(record.replay.config, record.replay.seed)
        for action_index, action in enumerate(record.replay.actions):
            actor = (
                state.active_player if state.phase is Phase.PLAY else state.active_player.other()
            )
            if record.seats[0 if actor is PlayerId.PLAYER_ONE else 1].agent_id == CONTROL_ID:
                observation = observe(state, actor)
                oracle = independent_public_forced_win_oracle(
                    observation,
                    observation.legal_actions,
                    authoritative_state=state,
                    exact_play_actions=(action,) if isinstance(action, PlayOfferAction) else (),
                )
                if oracle.forced_win_actions:
                    if action in oracle.forced_win_actions and len(oracle.forced_win_actions) > 1:
                        alternate = next(
                            candidate
                            for candidate in oracle.forced_win_actions
                            if candidate != action
                        )
                        selected = (record, action_index, alternate)
                    break
            state = apply_action(state, action)
        if selected is not None:
            break
    assert selected is not None
    control_record, endpoint_index, alternate = selected
    modified_actions = list(control_record.replay.actions)
    modified_actions[endpoint_index] = alternate
    scripts = {
        player: tuple(action for action in modified_actions if action.actor is player)
        for player in PlayerId
    }

    def scripted_spec(player: PlayerId) -> AgentSpec:
        agent_id = TREATMENT_ID if player is PlayerId.PLAYER_ONE else "prefix-regression-opponent"
        identity = Q0_RNG_IDENTITY if player is PlayerId.PLAYER_ONE else agent_id
        name = f"endpoint-{player.value}"
        actions = scripts[player]
        scripted = ScriptedAgent(actions, name=name)
        return AgentSpec(
            agent_id,
            scripted.config_to_data(),
            lambda actions=actions, name=name: ScriptedAgent(actions, name=name),
            rng_identity=identity,
        )

    treatment_record = run_game(
        GameSpec(
            "prefix-regression-treatment",
            "pair-000000-a-first",
            "pair-000000",
            control_record.replay.config,
            control_record.replay.seed,
            (
                scripted_spec(PlayerId.PLAYER_ONE),
                scripted_spec(PlayerId.PLAYER_TWO),
            ),
            (control_record.seats[0].seed, control_record.seats[1].seed),
            (
                control_record.seats[0].seed_derivation,
                control_record.seats[1].seed_derivation,
            ),
        )
    )

    comparison = compare_matched_action_prefixes(control_record, treatment_record)

    assert (
        control_record.replay.actions[endpoint_index]
        != treatment_record.replay.actions[endpoint_index]
    )
    assert comparison["prefix_aligned"] is True
    assert comparison["guaranteed_win_endpoint_found"] is True
    assert comparison["endpoint_classification"] == "both_converted"
    assert comparison["control_converted_at_endpoint"] is True
    assert comparison["treatment_converted_at_endpoint"] is True


def test_tiny_confirmation_smoke_is_resumable_immutable_and_validatable(
    tmp_path: Path,
) -> None:
    output = tmp_path / "confirmation-smoke"
    runner = Path("scripts/run_terminal_offense_confirmation_v1.py")
    validator = Path("scripts/validate_terminal_offense_confirmation_v1.py")
    command = (
        sys.executable,
        str(runner),
        "--output",
        str(output),
        "--pairs-per-family",
        "1",
        "--toy-agents",
    )

    first = subprocess.run(command, check=True, capture_output=True, text=True)
    first_result_bytes = (output / "result.json").read_bytes()
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert (output / "result.json").read_bytes() == first_result_bytes
    assert first.stdout
    assert second.stdout

    plan = json.loads((output / "plan.json").read_text())
    result = json.loads((output / "result.json").read_text())
    assert plan["frozen_default_design"]["pairs_per_family"] == DEFAULT_PAIRS_PER_FAMILY
    assert plan["frozen_default_design"]["total_games"] == 60_000
    assert plan["setup_holdout_scope"] == {
        "version": "all-completed-corpora-under-runs-v1",
        "declared_roots": ["runs"],
        "recursive": True,
        "excludes_current_output_subtree": True,
    }
    assert plan["execution_budget"]["claim_run_cutoff"] == "7h45m"
    assert plan["execution_budget"]["whole_step_compute_budget"] == "8h"
    assert plan["execution_budget"]["allowed_resume_retries"] == 1
    assert result["direct_treatment_vs_control"]["causal_matched_evidence"] is False
    assert result["direct_treatment_vs_control"]["used_for_practical_lift_claim"] is False
    assert (
        result["statistics"]["direct_treatment_vs_control_interpretation"][
            "used_for_structural_adoption"
        ]
        is False
    )
    assert result["setup_block_holdout"]["status"] == "passed"
    assert result["setup_block_holdout"]["overlap_count"] == 0
    assert len(result["setup_block_holdout"]["artifact_fingerprint"]) == 64
    assert result["games"]["total"] == 60
    assert (
        result["replay_audit"]["independent_production_oracle_agreement"]["counts"][
            "disagreement_decisions"
        ]
        == 0
    )
    treatment_counts = result["replay_audit"]["guaranteed_win_and_safety"]["treatment"]["counts"]
    assert treatment_counts["guaranteed_win_misses"] == 0
    assert treatment_counts["false_guaranteed_wins"] == 0
    assert (
        result["replay_audit"]["exact_max_logit_ties"]["by_arm"]["control"]["counts"][
            "selected_outside_exact_maxima"
        ]
        == 0
    )

    assert (
        result["replay_audit"]["exact_max_logit_ties"]["by_arm"]["control"]["counts"][
            "ties_with_forced_and_nonforced_maxima"
        ]
        > 0
    )
    prefix_counts = result["replay_audit"]["matched_action_prefixes"]["counts"]
    assert prefix_counts["metadata_mismatches"] == 0
    assert prefix_counts["pre_endpoint_prefix_mismatches"] == 0
    assert prefix_counts["treatment_endpoint_failures"] == 0

    validation = subprocess.run(
        (sys.executable, str(validator), str(output)),
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"status": "passed"' in validation.stdout

    changed = subprocess.run(
        (
            sys.executable,
            str(runner),
            "--output",
            str(output),
            "--pairs-per-family",
            "2",
            "--toy-agents",
        ),
        capture_output=True,
        text=True,
    )
    assert changed.returncode != 0
    assert "immutable artifact differs" in changed.stderr
