import subprocess
import sys
from pathlib import Path

import pytest


def test_game_stdout_is_directly_replayable(tmp_path: Path) -> None:
    record_path = tmp_path / "stdout.game.json"
    game = subprocess.run(
        [sys.executable, "-m", "agent_avenue", "game", "--seed", "17"],
        check=True,
        capture_output=True,
        text=True,
    )
    record_path.write_text(game.stdout)
    verified = subprocess.run(
        [sys.executable, "-m", "agent_avenue", "replay", str(record_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"verified": true' in verified.stdout
    assert '"format": "game_record"' in verified.stdout


def test_learned_checkpoint_agent_runs_without_recording_local_path(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from agent_avenue.learning import create_model, save_checkpoint

    checkpoint = tmp_path / "checkpoint"
    model = create_model(seed=3)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    saved = save_checkpoint(checkpoint, model, metrics={"fixture": True})
    game = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_avenue",
            "game",
            "--seed",
            "9",
            "--player-one",
            "learned",
            "--player-one-checkpoint",
            str(checkpoint),
            "--player-two",
            "random",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert saved.checkpoint_fingerprint in game.stdout
    assert str(checkpoint) not in game.stdout


def test_module_without_command_reports_helpful_error() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "agent_avenue"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "required: command" in result.stderr
