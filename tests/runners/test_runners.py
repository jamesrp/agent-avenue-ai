from dataclasses import replace

import pytest

from agent_avenue.agents import (
    DeterministicRandom,
    RandomAgent,
    RandomAgentConfig,
    ScriptedAgent,
    TerminalOffenseAgent,
    derive_seed,
)
from agent_avenue.engine import (
    GameConfig,
    PlayerId,
    PlayOfferAction,
    apply_action,
    legal_actions,
    new_game,
)
from agent_avenue.runners import (
    AgentController,
    AgentSpec,
    ArenaConfig,
    GameSession,
    GameSpec,
    HumanController,
    InvalidAgentActionError,
    advance_until_human_or_terminal,
    iter_games,
    paired_bootstrap_interval,
    run_arena,
    run_game,
    run_resumable_arena,
    schedule_arena,
    step_agent,
    wilson_interval,
)


def _random_spec(agent_id: str) -> AgentSpec:
    config = RandomAgentConfig()
    return AgentSpec(agent_id, config.to_data(), RandomAgent)


def test_resumable_arena_retains_verified_compressed_records(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = ArenaConfig("retained", _random_spec("a"), _random_spec("b"), 2, 71)
    first = run_resumable_arena(
        tmp_path / "records",
        config,
        generation=1,
        corpus_configuration={"stage": "fixture"},
    )
    second = run_resumable_arena(
        tmp_path / "records",
        config,
        generation=1,
        corpus_configuration={"stage": "fixture"},
    )
    assert first.records_manifest.corpus_fingerprint == second.records_manifest.corpus_fingerprint
    assert first.records_manifest.record_count == 4
    assert first.report.to_data()["wins"] == second.report.to_data()["wins"]
    assert (tmp_path / "records" / "manifest.json").is_file()
    assert (tmp_path / "records" / "games.jsonl.gz").is_file()


def test_agent_spec_defaults_rng_identity_without_changing_existing_rng_schedule() -> None:
    default_a = _random_spec("a")
    default_b = _random_spec("b")
    explicit_a = AgentSpec("a", default_a.config, RandomAgent, rng_identity="a")
    explicit_b = AgentSpec("b", default_b.config, RandomAgent, rng_identity="b")

    assert default_a.rng_identity == default_a.agent_id
    assert default_a.rng_domain == "agent:a"
    default_games = tuple(schedule_arena(ArenaConfig("compatible", default_a, default_b, 1, 99)))
    explicit_games = tuple(schedule_arena(ArenaConfig("compatible", explicit_a, explicit_b, 1, 99)))
    assert default_games == explicit_games
    assert default_games[0].agent_seeds == (
        derive_seed(99, "arena:pair:0:agent:a"),
        derive_seed(99, "arena:pair:0:agent:b"),
    )
    renamed_games = tuple(
        schedule_arena(ArenaConfig("compatible", _random_spec("x"), _random_spec("y"), 1, 99))
    )
    assert [game.setup_seed for game in renamed_games] == [
        game.setup_seed for game in default_games
    ]
    assert tuple(seat.rng_domain for seat in run_game(default_games[0]).seats) == (
        "agent:a",
        "agent:b",
    )


def test_agent_spec_rejects_empty_explicit_rng_identity() -> None:
    with pytest.raises(ValueError, match="rng_identity"):
        AgentSpec("a", RandomAgentConfig().to_data(), RandomAgent, rng_identity="")


def test_distinct_agents_can_share_rng_identity_across_arena_seat_swaps() -> None:
    random_agent = RandomAgent()
    offense_agent = TerminalOffenseAgent(RandomAgent())
    baseline = AgentSpec(
        "q0",
        random_agent.config.to_data(),
        lambda: random_agent,
        rng_identity="q0-matched-control",
    )
    treatment = AgentSpec(
        "q0-terminal-offense",
        offense_agent.config_to_data(),
        lambda: offense_agent,
        rng_identity="q0-matched-control",
    )

    first, second = tuple(schedule_arena(ArenaConfig("matched", baseline, treatment, 1, 1234)))
    expected_seed = derive_seed(1234, "arena:pair:0:agent:q0-matched-control")
    assert first.setup_seed == second.setup_seed
    assert first.agent_seeds == second.agent_seeds == (expected_seed, expected_seed)
    assert (
        first.agent_rng_domains
        == second.agent_rng_domains
        == (
            "agent:q0-matched-control",
            "agent:q0-matched-control",
        )
    )

    records = (run_game(first), run_game(second))
    assert {seat.agent_id for record in records for seat in record.seats} == {
        "q0",
        "q0-terminal-offense",
    }
    assert {(seat.seed, seat.rng_domain) for record in records for seat in record.seats} == {
        (expected_seed, "agent:q0-matched-control")
    }


def test_direct_game_supports_shared_rng_identity_for_distinct_agent_ids() -> None:
    first = AgentSpec(
        "control",
        RandomAgentConfig().to_data(),
        RandomAgent,
        rng_identity="matched",
    )
    second = AgentSpec(
        "treatment",
        RandomAgentConfig().to_data(),
        RandomAgent,
        rng_identity="matched",
    )
    record = run_game(
        GameSpec("direct", "shared", None, GameConfig(), 17, (first, second), (23, 23))
    )
    assert tuple((seat.seed, seat.rng_domain) for seat in record.seats) == (
        (23, "agent:matched"),
        (23, "agent:matched"),
    )


def test_seeded_games_reproduce_and_agent_randomness_is_separate_from_setup() -> None:
    spec = GameSpec(
        "run",
        "game",
        None,
        GameConfig(),
        42,
        (_random_spec("a"), _random_spec("b")),
        (100, 200),
    )
    first = run_game(spec)
    second = run_game(spec)
    assert first.replay.actions == second.replay.actions
    assert first.replay.final_fingerprint == second.replay.final_fingerprint
    assert first.replay.seed == 42
    assert all(seat.rng_algorithm for seat in first.seats)
    assert all(seat.rng_domain.startswith("agent:") for seat in first.seats)

    changed = run_game(replace(spec, agent_seeds=(101, 201)))
    assert changed.replay.seed == first.replay.seed
    assert new_game(seed=changed.replay.seed) == new_game(seed=first.replay.seed)


def test_agent_factory_configuration_must_match_recorded_configuration() -> None:
    claimed = RandomAgentConfig().to_data()
    claimed["version"] = "not-the-executed-version"
    spec = GameSpec(
        "run",
        "mismatch",
        None,
        GameConfig(),
        1,
        (AgentSpec("a", claimed, RandomAgent), _random_spec("b")),
        (1, 2),
    )
    with pytest.raises(ValueError, match="factory configuration"):
        run_game(spec)


def test_agent_without_normalized_configuration_is_rejected() -> None:
    class FirstLegalAgent:
        def choose_action(self, observation, decision, legal_actions, rng):  # type: ignore[no-untyped-def]
            return legal_actions[0]

    spec = GameSpec(
        "run",
        "missing-config",
        None,
        GameConfig(),
        1,
        (AgentSpec("a", {}, FirstLegalAgent), _random_spec("b")),
        (1, 2),
    )
    with pytest.raises(ValueError, match="configuration metadata"):
        run_game(spec)


def test_agent_spec_detaches_mutable_configuration() -> None:
    config = RandomAgentConfig().to_data()
    spec = AgentSpec("a", config, RandomAgent)
    config["version"] = "mutated"
    assert spec.config == RandomAgentConfig().to_data()


def test_scripted_complete_game_rejects_unused_actions() -> None:
    baseline_spec = GameSpec(
        "run",
        "baseline",
        None,
        GameConfig(),
        17,
        (_random_spec("random-a"), _random_spec("random-b")),
        (1, 2),
    )
    baseline = run_game(baseline_spec)
    scripts = tuple(
        tuple(action for action in baseline.replay.actions if action.actor is player)
        for player in PlayerId
    )

    def scripted_spec(actions, agent_id):  # type: ignore[no-untyped-def]
        config = ScriptedAgent(actions, name=agent_id).config_to_data()
        return AgentSpec(agent_id, config, lambda: ScriptedAgent(actions, name=agent_id))

    replayed = run_game(
        GameSpec(
            "run",
            "scripted",
            None,
            GameConfig(),
            17,
            (scripted_spec(scripts[0], "script-a"), scripted_spec(scripts[1], "script-b")),
            (3, 4),
        )
    )
    assert replayed.replay.actions == baseline.replay.actions

    extra = (*scripts[0], scripts[0][-1])
    with pytest.raises(ValueError, match="unused"):
        run_game(
            GameSpec(
                "run",
                "scripted-extra",
                None,
                GameConfig(),
                17,
                (scripted_spec(extra, "script-a"), scripted_spec(scripts[1], "script-b")),
                (3, 4),
            )
        )


def test_invalid_agent_output_is_rejected_without_mutating_state() -> None:
    class BadAgent:
        def choose_action(self, observation, decision, legal_actions, rng):  # type: ignore[no-untyped-def]
            action = legal_actions[0]
            assert isinstance(action, PlayOfferAction)
            return replace(action, revision=action.revision + 1)

    state = new_game(seed=3)
    session = GameSession(
        state,
        (
            AgentController(
                PlayerId.PLAYER_ONE, "bad", {}, 1, BadAgent(), DeterministicRandom(1, "bad")
            ),
            HumanController(PlayerId.PLAYER_TWO),
        ),
    )
    with pytest.raises(InvalidAgentActionError):
        step_agent(session)
    assert session.state is state


def test_resumable_runner_stops_at_human_and_can_finish_agent_turn() -> None:
    state = new_game(seed=7)
    session = GameSession(
        state,
        (
            HumanController(PlayerId.PLAYER_ONE),
            AgentController(
                PlayerId.PLAYER_TWO, "random", {}, 9, RandomAgent(), DeterministicRandom(9, "web")
            ),
        ),
    )
    assert advance_until_human_or_terminal(session).actions == ()
    session.state = apply_action(session.state, legal_actions(session.state)[0])
    result = advance_until_human_or_terminal(session)
    assert len(result.actions) == 2
    assert session.state.active_player is PlayerId.PLAYER_TWO
    assert session.state.phase.value == "recruit"


def test_batch_matches_individual_specs() -> None:
    specs = tuple(
        GameSpec(
            "run",
            f"g-{seed}",
            None,
            GameConfig(),
            seed,
            (_random_spec("a"), _random_spec("b")),
            (10, 20),
        )
        for seed in range(3)
    )
    batch = tuple(iter_games(specs))
    individual = tuple(run_game(spec) for spec in specs)
    assert [item.replay.actions for item in batch] == [item.replay.actions for item in individual]


def test_arena_schedule_balances_seats_and_preserves_paired_seeds() -> None:
    config = ArenaConfig("smoke", _random_spec("a"), _random_spec("b"), 3, 99)
    games = tuple(schedule_arena(config))
    assert len(games) == 6
    for first, second in zip(games[::2], games[1::2], strict=True):
        assert first.setup_seed == second.setup_seed
        assert first.seats == (config.agent_a, config.agent_b)
        assert second.seats == (config.agent_b, config.agent_a)
        assert first.agent_seeds == tuple(reversed(second.agent_seeds))


def test_arena_schedule_is_independent_of_run_label() -> None:
    first = ArenaConfig("label-one", _random_spec("a"), _random_spec("b"), 2, 99)
    second = ArenaConfig("label-two", _random_spec("a"), _random_spec("b"), 2, 99)
    first_games = tuple(schedule_arena(first))
    second_games = tuple(schedule_arena(second))
    assert [game.setup_seed for game in first_games] == [game.setup_seed for game in second_games]
    assert [game.agent_seeds for game in first_games] == [game.agent_seeds for game in second_games]


def test_paired_bootstrap_matches_declared_fixture() -> None:
    result = paired_bootstrap_interval((0, 0, 0, 1, 1, 1, 2, 2, 2, 2), 17)
    assert result.interval == (0.3, 0.8)
    assert result.point_estimate == 0.55
    assert paired_bootstrap_interval((0, 0), 17).interval == (0.0, 0.0)
    assert paired_bootstrap_interval((2, 2), 17).interval == (1.0, 1.0)
    with pytest.raises(ValueError):
        paired_bootstrap_interval((), 17)
    with pytest.raises(ValueError):
        paired_bootstrap_interval((3,), 17)


def test_arena_report_contains_reproducible_paired_outcomes() -> None:
    report = run_arena(ArenaConfig("paired", _random_spec("a"), _random_spec("b"), 2, 41))
    assert len(report.paired_seed_outcomes) == 2
    assert (
        sum(outcome.agent_a_wins for outcome in report.paired_seed_outcomes) == report.agent_a_wins
    )
    data = report.to_data()
    bootstrap = data["paired_bootstrap_confidence_interval_95"]
    assert isinstance(bootstrap, dict)
    assert bootstrap["method"] == "paired-percentile-bootstrap-v1"
    assert data["wilson_confidence_interval_95"] == data["confidence_interval_95"]


def test_wilson_interval_matches_known_fixture() -> None:
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038315)
    assert high == pytest.approx(0.5961685)
