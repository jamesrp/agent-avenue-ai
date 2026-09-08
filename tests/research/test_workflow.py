from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from agent_avenue.research import (
    WorkflowError,
    plan_fingerprint,
    request_stop,
    run_workflow,
    validate_plan,
)


def _task(
    task_id: str,
    code: str,
    *,
    depends_on: list[str] | None = None,
    outputs: list[dict[str, object]] | None = None,
    retries: int = 0,
    timeout: int = 10,
    resource_class: str = "light",
) -> dict[str, object]:
    return {
        "id": task_id,
        "kind": "experiment" if task_id == "experiment" else "analysis",
        "depends_on": depends_on or [],
        "command": [sys.executable, "-c", code],
        "working_directory": "{repo}",
        "timeout_seconds": timeout,
        "max_retries": retries,
        "resource_class": resource_class,
        "claim_generating": False,
        "freeze_source": False,
        "expected_outputs": outputs or [],
    }


def _plan(
    tasks: list[dict[str, object]], *, approved: bool = True, wall_time: int = 30
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "version": "bounded-research-workflow-v1",
        "cycle_id": "test-cycle",
        "approval_status": "approved" if approved else "proposed",
        "agreement": {
            "question": "Does the bounded workflow preserve evidence?",
            "hypotheses": ["Durable state prevents duplicate completed work."],
            "why_it_matters": "Unattended research must remain inspectable.",
            "frozen_evaluation": "Task outputs and terminal states are fixed before execution.",
            "reserved_decisions": "No follow-up cycle may start automatically.",
        },
        "limits": {
            "wall_time_seconds": wall_time,
            "max_concurrency": 2,
            "max_substantial_jobs": 1,
        },
        "tasks": tasks,
        "plan_fingerprint": "pending",
    }
    value["plan_fingerprint"] = plan_fingerprint(value)
    return value


def _write_plan(path: Path, plan: dict[str, Any]) -> Path:
    path.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    return path


def _wait_for(path: Path, predicate: Any, timeout: float = 10) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            state = json.loads(path.read_text())
            if predicate(state):
                return state
        time.sleep(0.02)
    raise AssertionError(f"condition not reached for {path}")


def test_plan_requires_approval_and_enforces_bounded_limits(tmp_path: Path) -> None:
    plan = _plan([_task("analysis", "pass")], approved=False)
    validate_plan(plan)
    path = _write_plan(tmp_path / "plan.json", plan)
    with pytest.raises(WorkflowError, match="approved plan"):
        run_workflow(path, tmp_path / "runtime")

    plan["approval_status"] = "approved"
    plan["limits"]["max_substantial_jobs"] = 2
    plan["plan_fingerprint"] = plan_fingerprint(plan)
    with pytest.raises(WorkflowError, match="may not exceed 1"):
        validate_plan(plan)


def test_workflow_runs_experiment_analysis_review_and_briefing_once(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    experiment = (
        "from pathlib import Path; import json; "
        f"p=Path({str(runtime / 'experiment.json')!r}); "
        "p.parent.mkdir(parents=True, exist_ok=True); "
        "p.write_text(json.dumps({'status':'completed','wins':3,'games':4}))"
    )
    analysis = (
        "from pathlib import Path; import json; "
        f"src=Path({str(runtime / 'experiment.json')!r}); "
        f"out=Path({str(runtime / 'analysis.json')!r}); "
        "d=json.loads(src.read_text()); "
        "out.write_text(json.dumps({'status':'completed','win_rate':d['wins']/d['games']}))"
    )
    review = (
        "from pathlib import Path; import json; "
        f"src=Path({str(runtime / 'analysis.json')!r}); "
        f"out=Path({str(runtime / 'review.json')!r}); "
        "d=json.loads(src.read_text()); "
        "out.write_text(json.dumps({'status':'completed','accepted':d['win_rate']==0.75}))"
    )
    briefing = (
        "from pathlib import Path; import json; "
        f"review=json.loads(Path({str(runtime / 'review.json')!r}).read_text()); "
        f"out=Path({str(runtime / 'BRIEFING.md')!r}); "
        "out.write_text('# Smoke briefing\\n\\nReview accepted: '"
        "+str(review['accepted'])+'\\n')"
    )
    tasks = [
        _task(
            "experiment",
            experiment,
            outputs=[
                {
                    "path": str(runtime / "experiment.json"),
                    "type": "json",
                    "json_contains": {"status": "completed"},
                }
            ],
            resource_class="substantial",
        ),
        _task(
            "analysis",
            analysis,
            depends_on=["experiment"],
            outputs=[
                {
                    "path": str(runtime / "analysis.json"),
                    "type": "json",
                    "json_contains": {"win_rate": 0.75},
                }
            ],
        ),
        _task(
            "review",
            review,
            depends_on=["analysis"],
            outputs=[
                {
                    "path": str(runtime / "review.json"),
                    "type": "json",
                    "json_contains": {"accepted": True},
                }
            ],
        ),
        _task(
            "briefing",
            briefing,
            depends_on=["review"],
            outputs=[{"path": str(runtime / "BRIEFING.md"), "type": "file", "min_bytes": 20}],
        ),
    ]
    path = _write_plan(tmp_path / "plan.json", _plan(tasks))
    first = run_workflow(path, runtime)
    second = run_workflow(path, runtime)
    state = json.loads(first.state_path.read_text())

    assert first.status == second.status == "completed"
    assert all(len(task["attempts"]) == 1 for task in state["tasks"].values())
    assert (runtime / "BRIEFING.md").read_text().startswith("# Smoke briefing")
    assert not (runtime / "next-cycle-started").exists()


def test_one_retry_then_block_dependents_without_stopping_independent_work(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    counter = runtime / "retry-count"
    retry_code = (
        "from pathlib import Path; import sys; "
        f"p=Path({str(counter)!r}); p.parent.mkdir(parents=True, exist_ok=True); "
        "n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n)); "
        f"out=Path({str(runtime / 'retry-ok.json')!r}); "
        'out.write_text(\'{"status":"completed"}\') if n==2 else None; '
        "sys.exit(0 if n==2 else 7)"
    )
    always_fail = "import sys; sys.exit(9)"
    independent = (
        f"from pathlib import Path; Path({str(runtime / 'independent.txt')!r}).write_text('done')"
    )
    tasks = [
        _task(
            "retrying",
            retry_code,
            retries=1,
            outputs=[
                {
                    "path": str(runtime / "retry-ok.json"),
                    "type": "json",
                    "json_contains": {"status": "completed"},
                }
            ],
        ),
        _task("permanent-failure", always_fail, retries=1),
        _task("dependent", "raise SystemExit('must not run')", depends_on=["permanent-failure"]),
        _task(
            "independent",
            independent,
            outputs=[{"path": str(runtime / "independent.txt"), "type": "file", "min_bytes": 4}],
        ),
    ]
    path = _write_plan(tmp_path / "plan.json", _plan(tasks))
    result = run_workflow(path, runtime)
    state = json.loads(result.state_path.read_text())

    assert result.status == "failed"
    assert state["tasks"]["retrying"]["status"] == "completed"
    assert len(state["tasks"]["retrying"]["attempts"]) == 2
    assert state["tasks"]["permanent-failure"]["status"] == "failed"
    assert len(state["tasks"]["permanent-failure"]["attempts"]) == 2
    assert state["tasks"]["dependent"]["status"] == "blocked"
    assert state["tasks"]["dependent"]["attempts"] == []
    assert state["tasks"]["independent"]["status"] == "completed"


def test_supervisor_interruption_resumes_without_duplicate_work(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    counter = runtime / "count"
    output = runtime / "done.json"
    code = (
        "from pathlib import Path; import time, json; "
        f"c=Path({str(counter)!r}); c.parent.mkdir(parents=True, exist_ok=True); "
        "n=int(c.read_text())+1 if c.exists() else 1; c.write_text(str(n)); time.sleep(0.5); "
        f"Path({str(output)!r}).write_text(json.dumps({{'status':'completed','count':n}}))"
    )
    task = _task(
        "analysis",
        code,
        outputs=[
            {
                "path": str(output),
                "type": "json",
                "json_contains": {"status": "completed", "count": 1},
            }
        ],
    )
    path = _write_plan(tmp_path / "plan.json", _plan([task]))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent_avenue.research",
            "run",
            str(path),
            "--runtime",
            str(runtime),
            "--no-notify",
        ],
        start_new_session=True,
    )
    state_path = runtime / "state.json"
    _wait_for(state_path, lambda state: state["tasks"]["analysis"]["status"] == "running")
    process.terminate()
    process.wait(timeout=5)

    result = run_workflow(path, runtime)
    state = json.loads(result.state_path.read_text())
    assert result.status == "completed"
    assert counter.read_text() == "1"
    assert len(state["tasks"]["analysis"]["attempts"]) == 1


def test_stop_halts_running_job_and_prevents_new_dispatch(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    first = _task("analysis", "import time; time.sleep(30)", timeout=60)
    second = _task(
        "briefing",
        f"from pathlib import Path; Path({str(runtime / 'should-not-exist')!r}).write_text('bad')",
        depends_on=["analysis"],
    )
    path = _write_plan(tmp_path / "plan.json", _plan([first, second]))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent_avenue.research",
            "run",
            str(path),
            "--runtime",
            str(runtime),
            "--no-notify",
        ],
        start_new_session=True,
    )
    state_path = runtime / "state.json"
    _wait_for(state_path, lambda state: state["tasks"]["analysis"]["status"] == "running")
    request_stop(path, runtime, reason="test stop")
    process.wait(timeout=10)
    state = json.loads(state_path.read_text())

    assert state["status"] == "stopped"
    assert state["tasks"]["analysis"]["status"] == "intentionally_stopped"
    assert state["tasks"]["briefing"]["status"] == "intentionally_stopped"
    assert not (runtime / "should-not-exist").exists()


def test_wall_time_budget_terminates_job_and_does_not_dispatch_followup(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    first = _task("analysis", "import time; time.sleep(30)", timeout=60)
    second = _task("briefing", "raise SystemExit('must not run')", depends_on=["analysis"])
    path = _write_plan(tmp_path / "plan.json", _plan([first, second], wall_time=1))
    result = run_workflow(path, runtime)
    state = json.loads(result.state_path.read_text())

    assert result.status == "budget_exhausted"
    assert state["tasks"]["analysis"]["status"] == "blocked"
    assert state["tasks"]["briefing"]["status"] == "blocked"
    assert state["tasks"]["briefing"]["attempts"] == []
