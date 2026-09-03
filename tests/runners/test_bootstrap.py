from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_avenue.runners.bootstrap import (
    BootstrapConfig,
    resolve_bootstrap_plan,
    run_bootstrap,
)


def test_tiny_bootstrap_runs_resumably_with_safety_diagnostics(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    config = BootstrapConfig(
        output=tmp_path / "q0",
        attempt_id="tiny-q0-a1",
        experiment_id="tiny-terminal-safety",
        root_seed=91,
        game_count=4,
        max_epochs=1,
        batch_size=32,
        patience=1,
        terminal_safety=True,
    )
    plan = resolve_bootstrap_plan(config)
    assert plan.to_data()["behavior_policy"]["policy_shield"] == "terminal-safety-v1"  # type: ignore[index]

    first = run_bootstrap(plan)
    second = run_bootstrap(plan)

    assert first.to_data() == second.to_data()
    assert (config.output / "corpus" / "manifest.json").is_file()
    assert (config.output / "dataset.npz").is_file()
    assert (config.output / "checkpoint" / "manifest.json").is_file()
    validation = json.loads((config.output / "validation.json").read_text())
    assert validation["checks"]["terminal_safety_audit_verified"] is True
    assert (
        validation["safety_diagnostics"]["training_corpus"]["counts"][
            "executed_avoidable_provable_losses"
        ]
        == 0
    )
    result = json.loads((config.output / "result.json").read_text())
    assert result["checkpoint_fingerprint"] == first.checkpoint_fingerprint
    assert result["claim_eligibility"]["eligible"] is False
