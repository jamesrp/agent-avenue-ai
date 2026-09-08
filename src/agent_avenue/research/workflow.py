"""Bounded, durable coordination for approved repository research cycles.

This module schedules trusted local commands. It deliberately does not replace the
project's experiment runners, decide scientific scope, or call models in a loop.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast

from agent_avenue.storage import inspect_source_identity, repository_root

WORKFLOW_SCHEMA_VERSION: Final = 1
WORKFLOW_VERSION: Final = "bounded-research-workflow-v1"
TERMINAL_TASK_STATES: Final = frozenset({"completed", "failed", "blocked", "intentionally_stopped"})
TERMINAL_WORKFLOW_STATES: Final = frozenset(
    {"completed", "failed", "blocked", "stopped", "budget_exhausted"}
)

TaskStatus = Literal["queued", "running", "completed", "failed", "blocked", "intentionally_stopped"]
WorkflowStatus = Literal[
    "queued", "running", "completed", "failed", "blocked", "stopped", "budget_exhausted"
]


class WorkflowError(ValueError):
    """Raised when a workflow plan or durable state is invalid."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(_canonical_json(value) + b"\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"unable to read JSON object {path}") from exc
    if not isinstance(value, dict):
        raise WorkflowError(f"{path} must contain a JSON object")
    return value


def plan_fingerprint(data: Mapping[str, object]) -> str:
    """Return the stable identity of a plan, excluding its declared fingerprint."""
    payload = {key: value for key, value in data.items() if key != "plan_fingerprint"}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkflowError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise WorkflowError(f"{label} must be an integer >= {minimum}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise WorkflowError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise WorkflowError(f"{label} must be an array")
    return value


def _validate_output(output: Mapping[str, object], label: str) -> None:
    _string(output.get("path"), f"{label}.path")
    output_type = output.get("type", "file")
    if output_type not in {"file", "directory", "json"}:
        raise WorkflowError(f"{label}.type must be file, directory, or json")
    if "min_bytes" in output:
        _integer(output["min_bytes"], f"{label}.min_bytes")
    if "json_contains" in output and not isinstance(output["json_contains"], Mapping):
        raise WorkflowError(f"{label}.json_contains must be an object")
    required_keys = output.get("required_keys", [])
    if not isinstance(required_keys, list) or any(
        not isinstance(item, str) or not item for item in required_keys
    ):
        raise WorkflowError(f"{label}.required_keys must contain non-empty strings")


def validate_plan(data: Mapping[str, object], *, require_approved: bool = False) -> None:
    """Validate the compact executable plan used after scientific approval."""
    if data.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
        raise WorkflowError(f"schema_version must be {WORKFLOW_SCHEMA_VERSION}")
    if data.get("version") != WORKFLOW_VERSION:
        raise WorkflowError(f"version must be {WORKFLOW_VERSION}")
    _string(data.get("cycle_id"), "cycle_id")
    approval = data.get("approval_status")
    if approval not in {"proposed", "approved"}:
        raise WorkflowError("approval_status must be proposed or approved")
    if require_approved and approval != "approved":
        raise WorkflowError("workflow execution requires an approved plan")
    declared = _string(data.get("plan_fingerprint"), "plan_fingerprint")
    actual = plan_fingerprint(data)
    if declared != actual:
        raise WorkflowError("plan_fingerprint does not match plan contents")

    agreement = _mapping(data.get("agreement"), "agreement")
    for key in ("question", "why_it_matters", "frozen_evaluation", "reserved_decisions"):
        _string(agreement.get(key), f"agreement.{key}")
    hypotheses = _sequence(agreement.get("hypotheses"), "agreement.hypotheses")
    if not hypotheses or any(not isinstance(item, str) or not item for item in hypotheses):
        raise WorkflowError("agreement.hypotheses must contain non-empty strings")

    limits = _mapping(data.get("limits"), "limits")
    _integer(limits.get("wall_time_seconds"), "limits.wall_time_seconds", minimum=1)
    concurrency = _integer(limits.get("max_concurrency"), "limits.max_concurrency", minimum=1)
    substantial = _integer(
        limits.get("max_substantial_jobs"), "limits.max_substantial_jobs", minimum=1
    )
    if concurrency > 3:
        raise WorkflowError("max_concurrency may not exceed the initial authorization of 3")
    if substantial > 1:
        raise WorkflowError("max_substantial_jobs may not exceed 1")

    raw_tasks = _sequence(data.get("tasks"), "tasks")
    if not raw_tasks:
        raise WorkflowError("tasks must not be empty")
    task_ids: set[str] = set()
    dependencies: dict[str, tuple[str, ...]] = {}
    for index, raw_task in enumerate(raw_tasks):
        task = _mapping(raw_task, f"tasks[{index}]")
        task_id = _string(task.get("id"), f"tasks[{index}].id")
        if task_id in task_ids:
            raise WorkflowError(f"duplicate task id: {task_id}")
        task_ids.add(task_id)
        kind = task.get("kind")
        if kind not in {"implementation", "check", "experiment", "analysis", "review", "briefing"}:
            raise WorkflowError(f"task {task_id} has unsupported kind")
        resource = task.get("resource_class", "light")
        if resource not in {"light", "substantial"}:
            raise WorkflowError(f"task {task_id} has unsupported resource_class")
        command = _sequence(task.get("command"), f"task {task_id}.command")
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise WorkflowError(f"task {task_id}.command must contain non-empty strings")
        raw_dependencies = _sequence(task.get("depends_on", []), f"task {task_id}.depends_on")
        if any(not isinstance(item, str) or not item for item in raw_dependencies):
            raise WorkflowError(f"task {task_id}.depends_on must contain task ids")
        dependencies[task_id] = tuple(cast(list[str], raw_dependencies))
        _integer(task.get("timeout_seconds"), f"task {task_id}.timeout_seconds", minimum=1)
        retries = _integer(task.get("max_retries", 0), f"task {task_id}.max_retries")
        if retries > 1:
            raise WorkflowError("setup policy permits at most one automatic retry")
        outputs = _sequence(task.get("expected_outputs", []), f"task {task_id}.expected_outputs")
        for output_index, raw_output in enumerate(outputs):
            _validate_output(
                _mapping(raw_output, f"task {task_id}.expected_outputs[{output_index}]"),
                f"task {task_id}.expected_outputs[{output_index}]",
            )
        if task.get("claim_generating", False) and kind != "experiment":
            raise WorkflowError("claim_generating is only valid for experiment tasks")
        if task.get("claim_generating", False) and not task.get("freeze_source", False):
            raise WorkflowError("claim-generating experiments must freeze their clean source")
        if task.get("claim_relevant", False) and not task.get("freeze_source", False):
            raise WorkflowError("claim-relevant tasks must freeze their clean source")
        if task.get("claim_generating", False):
            if not task.get("resume_safe", False):
                raise WorkflowError("claim-generating retries require a resume-safe runner")
            manifests = [
                cast(Mapping[str, object], output)
                for output in outputs
                if isinstance(output, Mapping)
                and output.get("type") == "json"
                and output.get("required_keys")
            ]
            if not manifests:
                raise WorkflowError(
                    "claim-generating tasks require a JSON manifest with required_keys"
                )

    for task_id, required in dependencies.items():
        unknown = set(required) - task_ids
        if unknown:
            raise WorkflowError(f"task {task_id} has unknown dependencies: {sorted(unknown)}")
        if task_id in required:
            raise WorkflowError(f"task {task_id} depends on itself")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise WorkflowError("task dependency graph contains a cycle")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in dependencies[task_id]:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in task_ids:
        visit(task_id)


def load_plan(path: Path, *, require_approved: bool = False) -> dict[str, object]:
    data = _read_object(path)
    validate_plan(data, require_approved=require_approved)
    return data


def _expand(value: str, *, runtime: Path, repo: Path, task_id: str) -> str:
    return (
        value.replace("{runtime}", str(runtime))
        .replace("{repo}", str(repo))
        .replace("{task_id}", task_id)
    )


def _json_contains(actual: object, expected: object) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _json_contains(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _json_contains(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def _resolved_output_path(
    output: Mapping[str, object], *, task_id: str, runtime: Path, repo: Path
) -> Path:
    raw_path = cast(str, output["path"])
    path = Path(_expand(raw_path, runtime=runtime, repo=repo, task_id=task_id)).resolve()
    if not path.is_relative_to(runtime.resolve()):
        raise WorkflowError(f"declared output escapes workflow runtime: {path}")
    return path


def validate_expected_outputs(
    task: Mapping[str, object], *, runtime: Path, repo: Path
) -> tuple[bool, list[str]]:
    """Validate declared completion evidence without interpreting experiment science."""
    task_id = _string(task.get("id"), "task.id")
    problems: list[str] = []
    outputs = cast(list[object], task.get("expected_outputs", []))
    for raw_output in outputs:
        output = cast(Mapping[str, object], raw_output)
        path = _resolved_output_path(output, task_id=task_id, runtime=runtime, repo=repo)
        output_type = output.get("type", "file")
        if output_type in {"file", "json"} and not path.is_file():
            problems.append(f"missing file: {path}")
            continue
        if output_type == "directory" and not path.is_dir():
            problems.append(f"missing directory: {path}")
            continue
        minimum = cast(int, output.get("min_bytes", 0))
        if output_type != "directory" and path.stat().st_size < minimum:
            problems.append(f"output too small: {path}")
            continue
        if output_type == "json":
            try:
                actual = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                problems.append(f"invalid JSON: {path}")
                continue
            required_keys = cast(list[str], output.get("required_keys", []))
            if isinstance(actual, Mapping):
                missing_keys = [key for key in required_keys if key not in actual]
            else:
                missing_keys = required_keys
            if missing_keys:
                problems.append(f"JSON missing required keys {missing_keys}: {path}")
                continue
            expected = output.get("json_contains")
            if expected is not None and not _json_contains(actual, expected):
                problems.append(f"JSON content mismatch: {path}")
    return not problems, problems


def _validate_runtime_boundaries(plan: Mapping[str, object], *, runtime: Path, repo: Path) -> None:
    for task in cast(list[Mapping[str, object]], plan["tasks"]):
        task_id = cast(str, task["id"])
        workdir = Path(
            _expand(
                cast(str, task.get("working_directory", "{repo}")),
                runtime=runtime,
                repo=repo,
                task_id=task_id,
            )
        ).resolve()
        if not workdir.is_relative_to(repo.resolve()):
            raise WorkflowError(f"task {task_id} working directory escapes repository: {workdir}")
        for raw_output in cast(list[object], task.get("expected_outputs", [])):
            _resolved_output_path(
                cast(Mapping[str, object], raw_output),
                task_id=task_id,
                runtime=runtime,
                repo=repo,
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_output_evidence(
    task: Mapping[str, object], *, runtime: Path, repo: Path
) -> list[dict[str, object]]:
    task_id = cast(str, task["id"])
    evidence: list[dict[str, object]] = []
    for raw_output in cast(list[object], task.get("expected_outputs", [])):
        output = cast(Mapping[str, object], raw_output)
        path = _resolved_output_path(output, task_id=task_id, runtime=runtime, repo=repo)
        item: dict[str, object] = {
            "path": str(path),
            "type": output.get("type", "file"),
        }
        if path.is_file():
            item["size"] = path.stat().st_size
            item["sha256"] = _sha256_file(path)
        elif path.is_dir():
            item["entry_count"] = sum(1 for _ in path.iterdir())
        evidence.append(item)
    return evidence


def _validate_completed_evidence(
    plan: Mapping[str, object], state: dict[str, object], runtime: Path, repo: Path
) -> None:
    definitions = _task_map(plan)
    for task_id, task_state in _state_tasks(state).items():
        if task_state["status"] != "completed":
            continue
        valid, problems = validate_expected_outputs(
            definitions[task_id], runtime=runtime, repo=repo
        )
        current = _collect_output_evidence(definitions[task_id], runtime=runtime, repo=repo)
        recorded = task_state.get("output_evidence")
        if not valid or recorded != current:
            task_state["status"] = "failed"
            state["finished_at"] = None
            task_state["last_error"] = (
                f"completed output evidence changed: validation={problems}; "
                f"recorded={recorded}; current={current}"
            )
            _event(state, "completed_evidence_changed", task_id=task_id)


def _initial_state(plan: Mapping[str, object], plan_path: Path) -> dict[str, object]:
    now = _now()
    tasks: dict[str, object] = {
        cast(str, task["id"]): {
            "status": "queued",
            "attempts": [],
            "retries_used": 0,
            "last_error": None,
            "source_snapshot": None,
            "output_evidence": [],
        }
        for task in cast(list[Mapping[str, object]], plan["tasks"])
    }
    return {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "version": WORKFLOW_VERSION,
        "cycle_id": plan["cycle_id"],
        "plan_path": str(plan_path.resolve()),
        "plan_fingerprint": plan["plan_fingerprint"],
        "status": "queued",
        "created_at": now,
        "started_at": None,
        "updated_at": now,
        "finished_at": None,
        "stop_requested": False,
        "stop_reason": None,
        "budget_exhausted": False,
        "tasks": tasks,
        "events": [],
        "notification": {"status": "not_configured", "conversation_id": None},
    }


def _event(state: dict[str, object], event: str, **details: object) -> None:
    events = cast(list[object], state["events"])
    events.append({"at": _now(), "event": event, **details})
    state["updated_at"] = _now()


def _task_map(plan: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {cast(str, task["id"]): task for task in cast(list[Mapping[str, object]], plan["tasks"])}


def _state_tasks(state: dict[str, object]) -> dict[str, dict[str, object]]:
    return cast(dict[str, dict[str, object]], state["tasks"])


def _pid_alive(pid: int) -> bool:
    try:
        status = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    except OSError:
        status = ""
    if status:
        closing = status.rfind(")")
        if closing >= 0 and status[closing + 2 : closing + 3] == "Z":
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _source_data() -> dict[str, object]:
    return inspect_source_identity().to_data()


def _prepare_runtime(
    plan: Mapping[str, object], plan_path: Path, runtime: Path
) -> dict[str, object]:
    runtime.mkdir(parents=True, exist_ok=True)
    state_path = runtime / "state.json"
    if state_path.exists():
        state = _read_object(state_path)
        if (
            state.get("plan_fingerprint") != plan["plan_fingerprint"]
            or state.get("cycle_id") != plan["cycle_id"]
        ):
            raise WorkflowError("runtime state belongs to a different immutable plan")
        return state
    state = _initial_state(plan, plan_path)
    _atomic_json(state_path, state)
    return state


def _attempt_exit_path(runtime: Path, task_id: str, attempt_number: int) -> Path:
    return runtime / "tasks" / task_id / f"attempt-{attempt_number}" / "exit.json"


def _recover_running_tasks(
    plan: Mapping[str, object], state: dict[str, object], runtime: Path, repo: Path
) -> None:
    tasks = _task_map(plan)
    for task_id, task_state in _state_tasks(state).items():
        if task_state["status"] != "running":
            continue
        attempts = cast(list[dict[str, object]], task_state["attempts"])
        if not attempts:
            raise WorkflowError(f"running task {task_id} has no attempt record")
        latest = attempts[-1]
        attempt_number = cast(int, latest["number"])
        exit_path = _attempt_exit_path(runtime, task_id, attempt_number)
        if exit_path.is_file():
            _finish_attempt(tasks[task_id], task_state, exit_path, runtime, repo, state)
            continue
        pid = latest.get("wrapper_pid")
        if isinstance(pid, int) and _pid_alive(pid):
            continue
        task = tasks[task_id]
        has_declared_outputs = bool(task.get("expected_outputs", []))
        valid, problems = validate_expected_outputs(task, runtime=runtime, repo=repo)
        if has_declared_outputs and valid:
            task_state["status"] = "completed"
            task_state["output_evidence"] = _collect_output_evidence(
                task, runtime=runtime, repo=repo
            )
            latest["recovered_without_exit"] = True
            latest["finished_at"] = _now()
            _event(state, "task_recovered_from_outputs", task_id=task_id)
        else:
            task_state["status"] = "queued"
            latest["interrupted"] = True
            latest["finished_at"] = _now()
            latest["output_problems"] = problems
            _event(state, "task_interrupted", task_id=task_id)


def _finish_attempt(
    task: Mapping[str, object],
    task_state: dict[str, object],
    exit_path: Path,
    runtime: Path,
    repo: Path,
    state: dict[str, object],
) -> None:
    result = _read_object(exit_path)
    attempts = cast(list[dict[str, object]], task_state["attempts"])
    latest = attempts[-1]
    latest["finished_at"] = result.get("finished_at")
    latest["exit_code"] = result.get("exit_code")
    latest["timed_out"] = result.get("timed_out", False)
    latest["signal"] = result.get("signal")
    valid, problems = validate_expected_outputs(task, runtime=runtime, repo=repo)
    latest["output_problems"] = problems
    exit_code = result.get("exit_code")
    task_id = cast(str, task["id"])
    if state.get("stop_requested"):
        task_state["status"] = "intentionally_stopped"
        task_state["last_error"] = "workflow stop requested"
        _event(state, "task_stopped", task_id=task_id)
    elif state.get("budget_exhausted"):
        task_state["status"] = "blocked"
        task_state["last_error"] = "workflow wall-time budget exhausted"
        _event(state, "task_blocked", task_id=task_id, reason="budget_exhausted")
    elif exit_code == 0 and valid:
        if task.get("freeze_source", False):
            snapshot = task_state.get("source_snapshot")
            if snapshot != _source_data():
                task_state["status"] = "failed"
                task_state["last_error"] = "source changed while frozen task was running"
                _event(state, "task_failed", task_id=task_id, reason=task_state["last_error"])
                return
        task_state["status"] = "completed"
        task_state["last_error"] = None
        task_state["output_evidence"] = _collect_output_evidence(task, runtime=runtime, repo=repo)
        _event(state, "task_completed", task_id=task_id)
    else:
        retries_used = cast(int, task_state["retries_used"])
        max_retries = cast(int, task.get("max_retries", 0))
        error = f"exit={exit_code}; outputs={problems}"
        task_state["last_error"] = error
        if retries_used < max_retries:
            task_state["retries_used"] = retries_used + 1
            task_state["status"] = "queued"
            _event(
                state,
                "task_retry_queued",
                task_id=task_id,
                retries_used=retries_used + 1,
                reason=error,
            )
        else:
            task_state["status"] = "failed"
            _event(state, "task_failed", task_id=task_id, reason=error)


def _signal_running_tasks(state: dict[str, object], *, reason: str) -> None:
    for task_id, task_state in _state_tasks(state).items():
        if task_state["status"] != "running":
            continue
        attempts = cast(list[dict[str, object]], task_state["attempts"])
        latest = attempts[-1] if attempts else None
        pid = latest.get("wrapper_pid") if latest else None
        marker = f"{reason}_signal_sent"
        if (
            latest is not None
            and not latest.get(marker)
            and isinstance(pid, int)
            and _pid_alive(pid)
        ):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGTERM)
                latest[marker] = True
                _event(
                    state,
                    "task_signal_sent",
                    task_id=task_id,
                    signal="SIGTERM",
                    reason=reason,
                )


def _block_dependents(plan: Mapping[str, object], state: dict[str, object]) -> None:
    task_states = _state_tasks(state)
    changed = True
    while changed:
        changed = False
        for task in cast(list[Mapping[str, object]], plan["tasks"]):
            task_id = cast(str, task["id"])
            task_state = task_states[task_id]
            if task_state["status"] != "queued":
                continue
            dependencies = cast(list[str], task.get("depends_on", []))
            bad = [
                dependency
                for dependency in dependencies
                if task_states[dependency]["status"]
                in {"failed", "blocked", "intentionally_stopped"}
            ]
            if bad:
                task_state["status"] = "blocked"
                task_state["last_error"] = f"blocked by dependencies: {bad}"
                _event(state, "task_blocked", task_id=task_id, dependencies=bad)
                changed = True


def _start_task(
    task: Mapping[str, object],
    task_state: dict[str, object],
    runtime: Path,
    repo: Path,
    state: dict[str, object],
) -> bool:
    task_id = cast(str, task["id"])
    if task.get("freeze_source", False):
        current_source = _source_data()
        if not current_source["tracked_tree_clean"]:
            task_state["status"] = "failed"
            task_state["last_error"] = "frozen task requires a clean tracked source tree"
            _event(state, "task_failed", task_id=task_id, reason=task_state["last_error"])
            return False
        snapshot = task_state.get("source_snapshot")
        if snapshot is None:
            task_state["source_snapshot"] = current_source
        elif snapshot != current_source:
            task_state["status"] = "failed"
            task_state["last_error"] = "source no longer matches frozen task snapshot"
            _event(state, "task_failed", task_id=task_id, reason=task_state["last_error"])
            return False

    attempts = cast(list[dict[str, object]], task_state["attempts"])
    attempt_number = len(attempts) + 1
    attempt_dir = runtime / "tasks" / task_id / f"attempt-{attempt_number}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    command = [
        _expand(cast(str, item), runtime=runtime, repo=repo, task_id=task_id)
        for item in cast(list[object], task["command"])
    ]
    workdir = Path(
        _expand(
            cast(str, task.get("working_directory", "{repo}")),
            runtime=runtime,
            repo=repo,
            task_id=task_id,
        )
    ).resolve()
    if not workdir.is_relative_to(repo.resolve()):
        task_state["status"] = "failed"
        task_state["last_error"] = f"working directory escapes repository: {workdir}"
        _event(state, "task_failed", task_id=task_id, reason=task_state["last_error"])
        return False
    timeout = cast(int, task["timeout_seconds"])
    wrapper_command = [
        sys.executable,
        "-m",
        "agent_avenue.research.job",
        "--attempt-dir",
        str(attempt_dir),
        "--timeout-seconds",
        str(timeout),
        "--working-directory",
        str(workdir),
        "--",
        *command,
    ]
    process = subprocess.Popen(
        wrapper_command,
        cwd=repo,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    attempts.append(
        {
            "number": attempt_number,
            "started_at": _now(),
            "wrapper_pid": process.pid,
            "command": command,
            "working_directory": str(workdir),
            "exit_code": None,
        }
    )
    task_state["status"] = "running"
    _event(state, "task_started", task_id=task_id, attempt=attempt_number, pid=process.pid)
    return True


def _elapsed_seconds(state: Mapping[str, object]) -> float:
    started = state.get("started_at")
    if not isinstance(started, str):
        return 0.0
    return (datetime.now(UTC) - datetime.fromisoformat(started)).total_seconds()


def _update_workflow_status(plan: Mapping[str, object], state: dict[str, object]) -> None:
    statuses = [task["status"] for task in _state_tasks(state).values()]
    if state.get("stop_requested") and not any(status == "running" for status in statuses):
        state["status"] = "stopped"
    elif all(status == "completed" for status in statuses):
        state["status"] = "completed"
    elif any(status == "failed" for status in statuses) and all(
        status in TERMINAL_TASK_STATES for status in statuses
    ):
        state["status"] = "failed"
    elif any(status == "blocked" for status in statuses) and all(
        status in TERMINAL_TASK_STATES for status in statuses
    ):
        state["status"] = "blocked"
    elif any(status == "running" for status in statuses):
        state["status"] = "running"
    else:
        state["status"] = "queued"
    if state["status"] in TERMINAL_WORKFLOW_STATES and state.get("finished_at") is None:
        state["finished_at"] = _now()
        _event(state, "workflow_finished", status=state["status"])


def _notify_once(state: dict[str, object], runtime: Path, conversation_id: str | None) -> None:
    if conversation_id is None or state["status"] not in TERMINAL_WORKFLOW_STATES:
        return
    notification = cast(dict[str, object], state["notification"])
    if notification.get("status") in {"sending", "sent"}:
        return
    notification.update(
        {"status": "sending", "conversation_id": conversation_id, "started_at": _now()}
    )
    _atomic_json(runtime / "state.json", state)
    message = (
        f"Research workflow {state['cycle_id']} reached terminal status {state['status']}. "
        f"Resume from durable state at {runtime / 'state.json'}; "
        "validate artifacts before briefing."
    )
    completed = subprocess.run(
        ["shelley", "client", "chat", "-c", conversation_id, "-p", message],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    notification.update(
        {
            "status": "sent" if completed.returncode == 0 else "failed",
            "finished_at": _now(),
            "exit_code": completed.returncode,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
        }
    )
    _event(state, "notification_finished", status=notification["status"])


def _control_path(runtime: Path) -> Path:
    return runtime / "control.json"


def _read_stop_control(runtime: Path) -> dict[str, object] | None:
    path = _control_path(runtime)
    if not path.exists():
        return None
    return _read_object(path)


@dataclass(frozen=True, slots=True)
class RunResult:
    state_path: Path
    status: str


def run_workflow(
    plan_path: Path,
    runtime: Path,
    *,
    poll_seconds: float = 0.1,
    conversation_id: str | None = None,
    clear_stop: bool = False,
) -> RunResult:
    """Run or resume an approved plan until it reaches a terminal state."""
    plan = load_plan(plan_path, require_approved=True)
    repo = repository_root()
    runtime = runtime.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    _validate_runtime_boundaries(plan, runtime=runtime, repo=repo)
    lock = (runtime / ".workflow.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise WorkflowError("workflow is already supervised for this runtime") from exc
    try:
        state = _prepare_runtime(plan, plan_path, runtime)
        if clear_stop:
            _atomic_json(
                _control_path(runtime),
                {"stop_requested": False, "reason": None, "updated_at": _now()},
            )
            state["stop_requested"] = False
            state["stop_reason"] = None
            state["budget_exhausted"] = False
            state["finished_at"] = None
            for task_state in _state_tasks(state).values():
                if task_state["status"] == "intentionally_stopped":
                    task_state["status"] = "queued"
            _event(state, "stop_cleared")
        control = _read_stop_control(runtime)
        if control and control.get("stop_requested"):
            if not state.get("stop_requested"):
                _event(state, "stop_observed", reason=control.get("reason"))
            state["stop_requested"] = True
            state["stop_reason"] = control.get("reason")
            for task_id, task_state in _state_tasks(state).items():
                if task_state["status"] == "queued":
                    task_state["status"] = "intentionally_stopped"
                    task_state["last_error"] = "workflow stop requested before dispatch"
                    _event(state, "task_stopped", task_id=task_id)
        _validate_completed_evidence(plan, state, runtime, repo)
        _block_dependents(plan, state)
        _update_workflow_status(plan, state)
        if state["status"] in TERMINAL_WORKFLOW_STATES and not clear_stop:
            _notify_once(state, runtime, conversation_id)
            _atomic_json(runtime / "state.json", state)
            return RunResult(runtime / "state.json", state["status"])
        if state.get("started_at") is None:
            state["started_at"] = _now()
            _event(state, "workflow_started")
        state["status"] = "running"
        _atomic_json(runtime / "state.json", state)
        task_definitions = _task_map(plan)
        limits = cast(Mapping[str, object], plan["limits"])
        wall_time = cast(int, limits["wall_time_seconds"])
        max_concurrency = cast(int, limits["max_concurrency"])
        max_substantial = cast(int, limits["max_substantial_jobs"])

        while True:
            disk_state = _read_object(runtime / "state.json")
            control = _read_stop_control(runtime)
            if control and control.get("stop_requested"):
                if not state.get("stop_requested"):
                    _event(state, "stop_observed", reason=control.get("reason"))
                state["stop_requested"] = True
                state["stop_reason"] = control.get("reason")
            elif disk_state.get("stop_requested"):
                state["stop_requested"] = True
                state["stop_reason"] = disk_state.get("stop_reason")
            _recover_running_tasks(plan, state, runtime, repo)
            _validate_completed_evidence(plan, state, runtime, repo)
            _block_dependents(plan, state)
            task_states = _state_tasks(state)

            if state.get("stop_requested"):
                _signal_running_tasks(state, reason="stop")
                for task_id, task_state in task_states.items():
                    if task_state["status"] == "queued":
                        task_state["status"] = "intentionally_stopped"
                        task_state["last_error"] = "workflow stop requested before dispatch"
                        _event(state, "task_stopped", task_id=task_id)
            elif _elapsed_seconds(state) >= wall_time:
                state["budget_exhausted"] = True
                for task_id, task_state in task_states.items():
                    if task_state["status"] == "queued":
                        task_state["status"] = "blocked"
                        task_state["last_error"] = "workflow wall-time budget exhausted"
                        _event(state, "task_blocked", task_id=task_id, reason="budget_exhausted")
                    elif task_state["status"] == "running":
                        _signal_running_tasks(state, reason="budget")
                if not any(task["status"] == "running" for task in task_states.values()):
                    state["status"] = "budget_exhausted"
                    if state.get("finished_at") is None:
                        state["finished_at"] = _now()
                        _event(state, "workflow_finished", status="budget_exhausted")
            else:
                running_ids = [
                    task_id
                    for task_id, task_state in task_states.items()
                    if task_state["status"] == "running"
                ]
                substantial_running = sum(
                    task_definitions[task_id].get("resource_class", "light") == "substantial"
                    for task_id in running_ids
                )
                available = max_concurrency - len(running_ids)
                if available > 0:
                    for task in cast(list[Mapping[str, object]], plan["tasks"]):
                        if available <= 0:
                            break
                        task_id = cast(str, task["id"])
                        task_state = task_states[task_id]
                        if task_state["status"] != "queued":
                            continue
                        dependencies = cast(list[str], task.get("depends_on", []))
                        if not all(
                            task_states[item]["status"] == "completed" for item in dependencies
                        ):
                            continue
                        is_substantial = task.get("resource_class", "light") == "substantial"
                        if is_substantial and substantial_running >= max_substantial:
                            continue
                        if _start_task(task, task_state, runtime, repo, state):
                            available -= 1
                            if is_substantial:
                                substantial_running += 1

            _block_dependents(plan, state)
            if state.get("status") != "budget_exhausted":
                _update_workflow_status(plan, state)
            _atomic_json(runtime / "state.json", state)
            if state["status"] in TERMINAL_WORKFLOW_STATES:
                _notify_once(state, runtime, conversation_id)
                _atomic_json(runtime / "state.json", state)
                return RunResult(runtime / "state.json", state["status"])
            time.sleep(poll_seconds)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def workflow_status(plan_path: Path, runtime: Path) -> dict[str, object]:
    plan = load_plan(plan_path)
    state = _prepare_runtime(plan, plan_path, runtime.resolve())
    return state


def request_stop(plan_path: Path, runtime: Path, *, reason: str) -> dict[str, object]:
    plan = load_plan(plan_path)
    runtime = runtime.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    control_path = _control_path(runtime)
    _atomic_json(
        control_path,
        {"stop_requested": True, "reason": reason, "updated_at": _now()},
    )
    state_path = runtime / "state.json"
    state = _read_object(state_path) if state_path.exists() else _initial_state(plan, plan_path)
    for task_state in _state_tasks(state).values():
        if task_state["status"] != "running":
            continue
        attempts = cast(list[dict[str, object]], task_state["attempts"])
        pid = attempts[-1].get("wrapper_pid") if attempts else None
        if isinstance(pid, int) and _pid_alive(pid):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGTERM)
    response = dict(state)
    response["stop_requested"] = True
    response["stop_reason"] = reason
    response["control_path"] = str(control_path)
    return response
