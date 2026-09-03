from __future__ import annotations

from agent_avenue.agents import EpsilonConfig
from agent_avenue.runners import (
    AgentSpec,
    GenerationConfig,
    generation_config_from_agent,
    planned_epsilon,
    schedule_generation,
)


def _agent(epsilon: EpsilonConfig) -> AgentSpec:
    config = {
        "version": "epsilon-wrapper-v1",
        "base": {
            "type": "learned_value",
            "checkpoint_fingerprint": "a" * 64,
            "tensor_digest": "b" * 64,
        },
        "epsilon": epsilon.to_data(),
    }
    return AgentSpec("frozen-incumbent", config, lambda: None)  # type: ignore[arg-type,return-value]


def _safe_agent(epsilon: EpsilonConfig) -> AgentSpec:
    base = _agent(epsilon)
    config = {
        "type": "terminal_safety",
        "version": "terminal-safety-v1",
        "base": dict(base.config),
        "fallback": "all-losing-preserve-complete-v1",
        "public_uncertainty": "public-remaining-multiset-enumeration-v1",
        "resolution_scope": "current-offer-resolution-only-v1",
        "terminal_evaluator": "engine-terminal-v1",
    }
    return AgentSpec("frozen-incumbent", config, lambda: None)  # type: ignore[arg-type,return-value]


def test_generation_schedule_is_deterministic_and_namespaced() -> None:
    epsilon = planned_epsilon(1)
    agent = _agent(epsilon)
    config = generation_config_from_agent(
        generation=1,
        run_id="g1",
        game_count=4,
        root_seed=42,
        agent=agent,
        epsilon=epsilon,
    )
    first = tuple(schedule_generation(config, agent))
    second = tuple(schedule_generation(config, agent))
    assert first == second
    assert [game.game_id for game in first] == [f"game-{index:06d}" for index in range(4)]
    assert len({game.setup_seed for game in first}) == 4
    assert all(game.seats == (agent, agent) for game in first)
    assert config.normalized()["parent_checkpoint_fingerprint"] == "a" * 64
    assert len(config.fingerprint) == 64


def test_terminal_safety_is_part_of_generation_identity_and_validation() -> None:
    epsilon = planned_epsilon(1)
    agent = _safe_agent(epsilon)
    config = generation_config_from_agent(
        generation=1,
        run_id="safe-g1",
        game_count=2,
        root_seed=42,
        agent=agent,
        epsilon=epsilon,
    )
    assert config.policy_shield == "terminal-safety-v1"
    assert config.normalized()["policy_shield"] == "terminal-safety-v1"
    assert len(tuple(schedule_generation(config, agent))) == 2

    unshielded = _agent(epsilon)
    try:
        tuple(schedule_generation(config, unshielded))
    except ValueError as exc:
        assert "policy shield" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unshielded agent matched shielded generation identity")


def test_generation_rejects_mismatched_behavior_identity() -> None:
    epsilon = EpsilonConfig(1, 10)
    agent = _agent(epsilon)
    config = GenerationConfig(1, "g1", 2, 7, "c" * 64, "b" * 64, epsilon)
    try:
        tuple(schedule_generation(config, agent))
    except ValueError as exc:
        assert "parent checkpoint" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mismatched behavior checkpoint was accepted")


def test_planned_epsilon_schedule_matches_declared_generations() -> None:
    assert [
        (planned_epsilon(index).numerator, planned_epsilon(index).denominator)
        for index in range(1, 5)
    ] == [
        (1, 10),
        (3, 40),
        (1, 20),
        (1, 40),
    ]
