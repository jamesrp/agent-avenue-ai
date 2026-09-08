"""Command-line interface for bounded research-cycle coordination."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .workflow import (
    load_plan,
    plan_fingerprint,
    request_stop,
    run_workflow,
    workflow_status,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a proposed or approved plan")
    validate.add_argument("plan", type=Path)

    fingerprint = subparsers.add_parser(
        "fingerprint", help="print the fingerprint for a plan draft"
    )
    fingerprint.add_argument("plan", type=Path)

    for name in ("run", "resume"):
        execute = subparsers.add_parser(name, help=f"{name} an approved workflow")
        execute.add_argument("plan", type=Path)
        execute.add_argument("--runtime", type=Path, required=True)
        execute.add_argument("--poll-seconds", type=float, default=0.1)
        execute.add_argument(
            "--notify-conversation",
            default=os.environ.get("SHELLEY_CONVERSATION_ID"),
            help="send one terminal-state continuation message through Shelley",
        )
        execute.add_argument("--no-notify", action="store_true")
        if name == "resume":
            execute.add_argument(
                "--clear-stop",
                action="store_true",
                help="explicitly resume intentionally stopped tasks",
            )

    status = subparsers.add_parser("status", help="print durable workflow state")
    status.add_argument("plan", type=Path)
    status.add_argument("--runtime", type=Path, required=True)

    stop = subparsers.add_parser("stop", help="stop new dispatch and terminate managed jobs")
    stop.add_argument("plan", type=Path)
    stop.add_argument("--runtime", type=Path, required=True)
    stop.add_argument("--reason", default="requested by research lead")
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, sort_keys=True, indent=2))


def main() -> None:
    args = _parser().parse_args()
    if args.command == "validate":
        plan = load_plan(args.plan)
        _print(
            {
                "valid": True,
                "cycle_id": plan["cycle_id"],
                "approval_status": plan["approval_status"],
                "plan_fingerprint": plan["plan_fingerprint"],
            }
        )
    elif args.command == "fingerprint":
        value = json.loads(args.plan.read_text())
        if not isinstance(value, dict):
            raise ValueError("plan draft must be a JSON object")
        _print({"plan_fingerprint": plan_fingerprint(value)})
    elif args.command in {"run", "resume"}:
        conversation_id = None if args.no_notify else args.notify_conversation
        result = run_workflow(
            args.plan,
            args.runtime,
            poll_seconds=args.poll_seconds,
            conversation_id=conversation_id,
            clear_stop=getattr(args, "clear_stop", False),
        )
        _print({"status": result.status, "state_path": str(result.state_path)})
    elif args.command == "status":
        _print(workflow_status(args.plan, args.runtime))
    else:
        _print(request_stop(args.plan, args.runtime, reason=args.reason))


if __name__ == "__main__":
    main()
