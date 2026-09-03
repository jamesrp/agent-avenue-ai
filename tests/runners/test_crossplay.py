from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_avenue.runners.crossplay import (
    CrossplayConfig,
    CrossplayError,
    resolve_crossplay_plan,
    run_crossplay,
)


def _checkpoint(path: Path, seed: int) -> Path:
    torch = pytest.importorskip("torch")
    from agent_avenue.learning import create_model, save_checkpoint

    model = create_model(seed=seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    save_checkpoint(path, model, metrics={"fixture": seed})
    return path


def test_crossplay_plan_enumerates_all_unordered_prior_policy_pairs(tmp_path: Path) -> None:
    candidate = _checkpoint(tmp_path / "candidate", 3)
    q0 = _checkpoint(tmp_path / "q0", 4)
    q1 = _checkpoint(tmp_path / "q1", 5)
    plan = resolve_crossplay_plan(
        CrossplayConfig(
            output=tmp_path / "crossplay",
            candidate_checkpoint=candidate,
            candidate_label="q2",
            generation=2,
            prior_checkpoints=(("q0", q0), ("q1", q1)),
            exclusion_corpora=(),
            root_seed=17,
            pair_count=2,
        )
    )

    assert [pair.pair_id for pair in plan.pairs] == [
        "00-heuristic-vs-heuristic",
        "01-heuristic-vs-q0",
        "02-heuristic-vs-q1",
        "03-q0-vs-q0",
        "04-q0-vs-q1",
        "05-q1-vs-q1",
    ]
    assert plan.to_data()["total_games"] == 24
    assert len({pair.master_seed for pair in plan.pairs}) == 6


def test_tiny_crossplay_runs_resumably_and_rejects_training_overlap(tmp_path: Path) -> None:
    candidate = _checkpoint(tmp_path / "candidate", 9)
    config = CrossplayConfig(
        output=tmp_path / "q0-crossplay",
        candidate_checkpoint=candidate,
        candidate_label="q0",
        generation=0,
        prior_checkpoints=(),
        exclusion_corpora=(),
        root_seed=23,
        pair_count=1,
    )
    plan = resolve_crossplay_plan(config)
    first_path = run_crossplay(plan)
    second_path = run_crossplay(plan)
    assert first_path == second_path

    report = json.loads(first_path.read_text())
    assert report["counts"] == {
        "matchups": 1,
        "paired_blocks_per_matchup": 1,
        "games": 2,
        "decisions": report["counts"]["decisions"],
    }
    assert report["heldout_verification"]["setup_overlap_count"] == 0
    assert report["promotion_evidence"] is False
    assert report["matchups"][0]["left"] == "heuristic"
    assert report["matchups"][0]["right"] == "heuristic"

    records = config.output / "records" / "00-heuristic-vs-heuristic"
    with pytest.raises(CrossplayError, match="overlaps"):
        resolve_crossplay_plan(
            CrossplayConfig(
                output=tmp_path / "overlap",
                candidate_checkpoint=candidate,
                candidate_label="q0",
                generation=0,
                prior_checkpoints=(),
                exclusion_corpora=(records,),
                root_seed=23,
                pair_count=1,
            )
        )
