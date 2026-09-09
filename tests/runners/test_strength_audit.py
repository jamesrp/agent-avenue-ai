from agent_avenue.agents import RandomAgent, RandomAgentConfig
from agent_avenue.runners import (
    AgentSpec,
    ArenaConfig,
    audit_public_forced_wins,
    audit_recruit_patterns,
    paired_policy_difference_interval,
    run_arena,
)
from agent_avenue.storage import GameRecord


def random_spec(agent_id: str) -> AgentSpec:
    config = RandomAgentConfig()
    return AgentSpec(agent_id, config.to_data(), lambda: RandomAgent(config))


def test_forced_win_and_recruit_audits_replay_complete_records() -> None:
    records: list[GameRecord] = []
    report = run_arena(
        ArenaConfig("strength-audit-test", random_spec("random-a"), random_spec("random-b"), 3, 17),
        records.append,
    )

    tactical = audit_public_forced_wins(
        records,
        source_label="test-records",
        example_agent_ids=frozenset({"random-a"}),
    )
    patterns = audit_recruit_patterns(
        records,
        target_agent_id="random-a",
        source_label="test-records",
    )

    assert tactical["record_count"] == report.total_games
    assert tactical["decision_count"] == sum(record.decision_count for record in records)
    assert set(tactical["by_agent"]) == {"random-a", "random-b"}
    assert patterns["target_agent_id"] == "random-a"
    assert patterns["overall"]["decisions"] > 0
    assert len(tactical["artifact_fingerprint"]) == 64
    assert len(patterns["artifact_fingerprint"]) == 64


def test_paired_policy_difference_uses_shared_blocks_deterministically() -> None:
    first = paired_policy_difference_interval(
        (2, 1, 2, 0),
        (1, 1, 0, 0),
        master_seed=91,
        domain="strength-audit:test-difference",
    )
    second = paired_policy_difference_interval(
        (2, 1, 2, 0),
        (1, 1, 0, 0),
        master_seed=91,
        domain="strength-audit:test-difference",
    )

    assert first == second
    assert first["point_estimate"] == 0.375
    assert first["interval"][0] <= 0.375 <= first["interval"][1]
