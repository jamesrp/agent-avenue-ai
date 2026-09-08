"""Durable subprocess wrapper used by the bounded research workflow."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

_TERMINATE_GRACE_SECONDS: Final = 10
_child: subprocess.Popen[bytes] | None = None
_received_signal: int | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(value, destination, sort_keys=True, separators=(",", ":"))
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _forward(signum: int, _frame: object) -> None:
    global _received_signal
    _received_signal = signum
    if _child is not None and _child.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(_child.pid, signum)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--working-directory", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    global _child
    args = _parser().parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing wrapped command")
    args.attempt_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = args.attempt_dir / "stdout.log"
    stderr_path = args.attempt_dir / "stderr.log"
    started_at = _now()
    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)
    timed_out = False
    exit_code: int
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        _child = subprocess.Popen(
            command,
            cwd=args.working_directory,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        if _received_signal is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(_child.pid, _received_signal)
        _atomic_json(
            args.attempt_dir / "started.json",
            {
                "started_at": started_at,
                "wrapper_pid": os.getpid(),
                "child_pid": _child.pid,
                "command": command,
                "working_directory": str(args.working_directory),
            },
        )
        try:
            exit_code = _child.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(_child.pid, signal.SIGTERM)
            try:
                exit_code = _child.wait(timeout=_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(_child.pid, signal.SIGKILL)
                exit_code = _child.wait()
    _atomic_json(
        args.attempt_dir / "exit.json",
        {
            "started_at": started_at,
            "finished_at": _now(),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "signal": signal.Signals(_received_signal).name if _received_signal else None,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()
