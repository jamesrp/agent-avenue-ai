from __future__ import annotations

from agent_avenue.agents import (
    RandomAgent,
    RandomAgentConfig,
    TerminalSafetyAgent,
)
from agent_avenue.runners import AgentSpec, ArenaConfig, run_arena, schedule_arena
from agent_avenue.runners.game import run_game
from agent_avenue.runners.safety_audit import audit_terminal_safety


def test_safety_audit_is_deterministic_and_enforces_shield_contract() -> None:
    random_config = RandomAgentConfig()
    shielded = TerminalSafetyAgent(RandomAgent(random_config))
    shielded_spec = AgentSpec("shielded-random", shielded.config_to_data(), lambda: shielded)
    plain_spec = AgentSpec("plain-random", random_config.to_data(), RandomAgent)
    config = ArenaConfig("safety-audit", shielded_spec, plain_spec, 3, 77)
    records = tuple(run_game(spec) for spec in schedule_arena(config))

    first = audit_terminal_safety(records, source_corpus_fingerprint="fixture-corpus")
    second = audit_terminal_safety(records, source_corpus_fingerprint="fixture-corpus")

    assert first == second
    assert first["record_count"] == 6
    assert first["counts"]["decisions"] == sum(record.decision_count for record in records)  # type: ignore[index]
    by_agent = first["by_agent"]
    assert by_agent["shielded-random"]["policy_shield"] == "terminal-safety-v1"  # type: ignore[index]
    assert (
        by_agent["shielded-random"]["counts"]["executed_avoidable_provable_losses"]  # type: ignore[index]
        == 0
    )
    assert isinstance(first["artifact_fingerprint"], str)


def test_arena_report_still_runs_with_shielded_agent() -> None:
    random_config = RandomAgentConfig()
    shielded = TerminalSafetyAgent(RandomAgent(random_config))
    report = run_arena(
        ArenaConfig(
            "shielded-arena",
            AgentSpec("shielded", shielded.config_to_data(), lambda: shielded),
            AgentSpec("plain", random_config.to_data(), RandomAgent),
            1,
            8,
        )
    )
    assert report.total_games == 2
