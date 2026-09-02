from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agent_avenue.runners import (
    IterationConfig,
    IterationError,
    resolve_iteration_plan,
    run_iteration,
)


def _checkpoint(path: Path) -> Path:
    torch = pytest.importorskip("torch")
    from agent_avenue.learning import create_model, save_checkpoint

    model = create_model(seed=11)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    save_checkpoint(path, model, metrics={"fixture": True})
    return path


def test_iteration_plan_separates_attempt_domains_without_writing(tmp_path: Path) -> None:
    incumbent = _checkpoint(tmp_path / "incumbent")
    first = resolve_iteration_plan(
        IterationConfig(tmp_path / "first", incumbent, 1, "attempt-1", 99, game_count=4)
    )
    repeated = resolve_iteration_plan(
        IterationConfig(tmp_path / "other-path", incumbent, 1, "attempt-1", 99, game_count=4)
    )
    retry = resolve_iteration_plan(
        IterationConfig(tmp_path / "retry", incumbent, 1, "attempt-2", 99, game_count=4)
    )
    assert first.fingerprint == repeated.fingerprint
    assert first.seeds == repeated.seeds
    assert first.fingerprint != retry.fingerprint
    assert set(first.seeds.values()).isdisjoint(retry.seeds.values())
    assert not (tmp_path / "first").exists()


def test_iteration_rejects_replaced_incumbent_after_planning(tmp_path: Path) -> None:
    incumbent = _checkpoint(tmp_path / "incumbent")
    plan = resolve_iteration_plan(
        IterationConfig(tmp_path / "iteration", incumbent, 1, "attempt-1", 5, game_count=2)
    )
    shutil.rmtree(incumbent)
    from agent_avenue.learning import create_model, save_checkpoint

    save_checkpoint(incumbent, create_model(seed=999), metrics={"replacement": True})
    with pytest.raises(IterationError, match="no longer matches"):
        run_iteration(plan)


def test_tiny_iteration_runs_end_to_end_and_retains_immutable_decision(tmp_path: Path) -> None:
    incumbent = _checkpoint(tmp_path / "incumbent")
    config = IterationConfig(
        output=tmp_path / "iteration",
        incumbent_checkpoint=incumbent,
        generation=1,
        attempt_id="tiny-attempt",
        root_seed=1234,
        game_count=4,
        primary_pairs=1,
        guardrail_pairs=1,
        confirmation_pairs=1,
        max_epochs=1,
        batch_size=32,
        patience=1,
    )
    plan = resolve_iteration_plan(config)
    first = run_iteration(plan)
    second = run_iteration(plan)
    assert first.to_data() == second.to_data()
    assert first.selected_role in {"candidate", "incumbent"}
    if first.selected_role == "incumbent":
        assert first.selected_checkpoint == plan.incumbent_fingerprint
    else:
        assert first.selected_checkpoint == first.candidate_checkpoint

    decision = json.loads(first.decision_path.read_text())
    assert decision["plan_fingerprint"] == plan.fingerprint
    assert decision["selected_checkpoint"] == first.selected_checkpoint
    assert decision["source"] == plan.source_identity
    assert decision["claim_eligibility"]["eligible"] is False
    assert (config.output / "corpus" / "manifest.json").is_file()
    assert (config.output / "dataset.npz").is_file()
    assert (config.output / "candidate" / "manifest.json").is_file()
    assert (config.output / "arenas" / "primary.json").is_file()
    assert (config.output / "validation.json").is_file()
    for stage in (
        "primary",
        "versus-random",
        "candidate-versus-heuristic",
        "incumbent-versus-heuristic",
    ):
        manifest = config.output / "arena-records" / stage / "manifest.json"
        records = config.output / "arena-records" / stage / "games.jsonl.gz"
        assert manifest.is_file()
        assert records.is_file()
