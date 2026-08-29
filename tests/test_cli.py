import subprocess
import sys
from pathlib import Path


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


def test_module_without_command_reports_helpful_error() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "agent_avenue"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "required: command" in result.stderr
