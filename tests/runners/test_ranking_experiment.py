from pathlib import Path

from agent_avenue.agents import GreedyHeuristicAgent, GreedyHeuristicConfig
from agent_avenue.engine import GameConfig
from agent_avenue.learning import create_model, materialize_dataset, save_checkpoint, save_dataset
from agent_avenue.runners import AgentSpec, GameSpec, run_game
from agent_avenue.runners.ranking_experiment import (
    HISTORICAL_Q1_MC_SEED,
    RankingExperimentConfig,
    _nested_interval,
    resolve_ranking_experiment_plan,
)
from agent_avenue.storage import write_corpus


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    heuristic = GreedyHeuristicConfig()
    spec = AgentSpec("heuristic", heuristic.to_data(), GreedyHeuristicAgent)
    records = tuple(
        run_game(
            GameSpec(
                "ranking-plan-test",
                f"game-{index}",
                None,
                GameConfig(),
                3000 + index,
                (spec, spec),
                (4000 + index, 5000 + index),
            )
        )
        for index in range(4)
    )
    corpus = tmp_path / "corpus"
    manifest = write_corpus(
        corpus,
        records,
        run_id="ranking-plan-test",
        behavior_policy="heuristic",
        root_seed=1,
    )
    dataset = materialize_dataset(
        records,
        split_seed=2,
        source_corpus_fingerprint=manifest.corpus_fingerprint,
    )
    dataset_path, _ = save_dataset(dataset, tmp_path / "dataset.npz")
    parent = tmp_path / "parent"
    historical = tmp_path / "historical"
    save_checkpoint(parent, create_model(seed=4), metrics={"fixture": "parent"})
    save_checkpoint(
        historical,
        create_model(seed=5),
        metrics={"fixture": "historical"},
        parent_checkpoint=save_checkpoint(
            tmp_path / "lineage-parent", create_model(seed=6), metrics={"fixture": True}
        ).checkpoint_fingerprint,
        generation=1,
    )
    return corpus, dataset_path, parent, historical


def test_ranking_plan_freezes_paired_seed_domains_and_inputs(tmp_path: Path) -> None:
    corpus, dataset, parent, historical = _inputs(tmp_path)
    config = RankingExperimentConfig(
        output=tmp_path / "output",
        corpus=corpus,
        mc_dataset=dataset,
        parent_checkpoint=parent,
        historical_q1_checkpoint=historical,
        replicates=2,
        direct_pairs=1,
        parent_pairs=1,
        heuristic_pairs=1,
        random_pairs=1,
        max_epochs=1,
        batch_size=8,
        patience=1,
    )
    first = resolve_ranking_experiment_plan(config)
    second = resolve_ranking_experiment_plan(config)

    assert first.fingerprint == second.fingerprint
    assert first.replicate_seeds == second.replicate_seeds
    assert first.replicate_seeds[0]["mc"] == HISTORICAL_Q1_MC_SEED
    assert first.replicate_seeds[0]["ranking"] != first.replicate_seeds[0]["mc"]
    assert len(set(first.arena_seeds.values())) == len(first.arena_seeds)
    assert first.input_identities["corpus"]["record_count"] == 4
    assert "historical_q1_tensor_mismatch" in first.claim_ineligibility_reasons


def test_nested_bootstrap_is_deterministic_and_keeps_replicate_unit() -> None:
    blocks = ((1.0, 1.0, 0.5), (0.5, 1.0, 0.5), (1.0, 0.5, 0.5))
    first = _nested_interval(blocks, seed=77, domain="test")
    second = _nested_interval(blocks, seed=77, domain="test")

    assert first == second
    assert first["replicate_count"] == 3
    assert first["unit"] == "training replicate and shared paired setup block"
    assert 0.5 <= first["point_estimate"] <= 1.0
