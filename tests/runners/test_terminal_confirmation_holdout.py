from pathlib import Path

import pytest

from agent_avenue.agents import RandomAgent, RandomAgentConfig
from agent_avenue.engine import GameConfig
from agent_avenue.runners import (
    AgentSpec,
    GameSpec,
    TerminalConfirmationError,
    family_setup_seeds,
    run_game,
    scan_prior_setup_blocks,
)
from agent_avenue.storage import write_corpus


def _random_spec(agent_id: str) -> AgentSpec:
    config = RandomAgentConfig()
    return AgentSpec(agent_id, config.to_data(), RandomAgent)


def _write_prior_corpus(directory: Path, setup_seed: int, *, game_id: str = "prior") -> None:
    record = run_game(
        GameSpec(
            "prior-run",
            game_id,
            None,
            GameConfig(),
            setup_seed,
            (_random_spec("prior-a"), _random_spec("prior-b")),
            (31, 47),
        )
    )
    write_corpus(
        directory,
        (record,),
        run_id="prior-run",
        behavior_policy="prior-fixture",
    )


def test_prior_setup_scan_reports_missing_roots_and_zero_overlap(tmp_path: Path) -> None:
    retained_root = tmp_path / "retained"
    _write_prior_corpus(retained_root / "ordinary-corpus", 17)
    current_output = retained_root / "current-output"
    overlap_seed = family_setup_seeds("seed-family-a", 1)[0]
    _write_prior_corpus(current_output / "records", overlap_seed, game_id="current")
    missing = tmp_path / "optional-missing"

    scan = scan_prior_setup_blocks(
        pair_count=1,
        excluded_roots=(retained_root, missing),
        current_output=current_output,
    )

    assert scan["status"] == "passed"
    assert scan["overlap_count"] == 0
    assert scan["scanned_corpus_count"] == 1
    assert scan["scanned_record_count"] == 1
    assert len(scan["artifact_fingerprint"]) == 64
    roots = {row["declared_root"]: row for row in scan["excluded_roots"]}
    assert roots[str(missing)]["status"] == "missing"
    assert roots[str(retained_root)]["current_output_excluded"] is True


def test_prior_setup_scan_detects_normalized_config_and_seed_overlap(tmp_path: Path) -> None:
    retained_root = tmp_path / "retained"
    overlap_seed = family_setup_seeds("seed-family-b", 1)[0]
    _write_prior_corpus(retained_root / "overlap-corpus", overlap_seed)

    scan = scan_prior_setup_blocks(
        pair_count=1,
        excluded_roots=(retained_root,),
        current_output=tmp_path / "new-output",
    )

    assert scan["status"] == "failed"
    assert scan["overlap_count"] == 1
    assert scan["overlap_examples"][0]["prior"]["setup_seed"] == overlap_seed
    assert scan["overlap_examples"][0]["current"]["setup_seed"] == overlap_seed


def test_prior_setup_scan_rejects_malformed_present_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "retained" / "broken"
    corpus.mkdir(parents=True)
    (corpus / "manifest.json").write_text('{"records_file":"games.jsonl.gz"}\n')

    with pytest.raises(TerminalConfirmationError, match="unable to load"):
        scan_prior_setup_blocks(
            pair_count=1,
            excluded_roots=(tmp_path / "retained",),
        )
