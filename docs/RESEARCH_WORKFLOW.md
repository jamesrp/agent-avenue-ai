# Bounded research workflow

**Status:** Implemented setup; no real cycle approved or running  
**Date:** September 8, 2026

This is small coordination glue around the repository's existing runners. It is not a second
experiment runner, an agent platform, or permission to continue research indefinitely.

## Operating agreement

Shelley is the user-facing research lead. At a substantive check-in, the user approves one compact,
versioned cycle agreement. The lead may then coordinate implementation, run only the authorized
commands, analyze evidence, obtain a fresh-context review, write one briefing, and stop for the next
user decision.

An agreement must freeze:

- question, hypotheses, importance, and scientific constraints;
- work packages, dependencies, deliverables, and completion criteria;
- evaluation blocks, primary metrics, guardrails, and prohibited claims;
- wall-time/compute limits, concurrency, agent-worker count, and retry count;
- allowed repair or confirmation branches; and
- decisions reserved for the user.

`approval_status` must be exactly `approved` before the executable plan runs. Its fingerprint covers
the full plan. The coordinator refuses a changed fingerprint and never creates a follow-up cycle.

## Reused project machinery

Claim-generating work continues to use the existing corpus, bootstrap, arena, crossplay, iteration,
checkpoint, and promotion runners. Those own semantic schedules, source/configuration identities,
resume rules, retained records, and scientific decisions. The workflow coordinator only orders
trusted commands, captures their exit status/logs, validates declared output evidence, and records
state.

`freeze_source: true` requires a clean tracked tree, records the source identity on first dispatch,
and rejects a changed source before accepting completion. A claim-generating task is invalid unless
source freezing is enabled. Existing experiment runners should still freeze their own normalized
plan and seeds; the coordinator is not a substitute for that check.

## Durable records

Committed inputs live under `research/cycles/`. Generated runtime state remains ignored under
`runs/research-cycles/<cycle-id>/`:

```text
state.json                         current state and append-only event list
tasks/<task>/attempt-N/started.json
                               exact command, cwd, wrapper PID, start time
tasks/<task>/attempt-N/stdout.log  command output
tasks/<task>/attempt-N/stderr.log  command errors
tasks/<task>/attempt-N/exit.json   actual exit code, timeout, signal, finish time
...                                declared experiment/analysis/review/briefing outputs
```

Task states are `queued`, `running`, `completed`, `failed`, `blocked`, or
`intentionally_stopped`. Workflow terminal states are `completed`, `failed`, `blocked`, `stopped`,
or `budget_exhausted`.

A task is never marked complete merely because it started. Completion requires exit code zero and
all declared outputs. Completed tasks are not relaunched. A task whose wrapper disappeared can be
recovered from valid declared outputs; otherwise it returns to the same idempotent/resumable command.
Failures receive at most the plan's bounded retry count (currently no more than one). Failed
prerequisites block dependent work but do not stop independent ready tasks.

## Commands

```bash
# Validate a proposal or approved executable plan.
uv run python -m agent_avenue.research validate research/cycles/<cycle>.json

# Run in the foreground (normally used for tests or short work).
uv run python -m agent_avenue.research run research/cycles/<cycle>.json \
  --runtime runs/research-cycles/<cycle> --no-notify

# Resume, inspect, or stop through Shelley.
uv run python -m agent_avenue.research resume research/cycles/<cycle>.json \
  --runtime runs/research-cycles/<cycle> --no-notify
uv run python -m agent_avenue.research status research/cycles/<cycle>.json \
  --runtime runs/research-cycles/<cycle>
uv run python -m agent_avenue.research stop research/cycles/<cycle>.json \
  --runtime runs/research-cycles/<cycle> --reason "user requested stop"
```

For a real unattended cycle, Shelley launches the same `run` command in a detached `tmux` session
and supplies `--notify-conversation "$SHELLEY_CONVERSATION_ID"`. Jobs are ordinary OS processes;
there is no model polling. At a terminal workflow state, one `shelley client chat -c ...` message
asks the same lead conversation to validate artifacts, conduct/collect the authorized fresh review,
and prepare the briefing. Notification failure is retained in `state.json` and can be retried by an
explicit resume.

## Stop and budget semantics

A stop request is durable, halts new dispatch, sends `SIGTERM` to the managed wrapper/process group,
and preserves logs and partial artifacts. Resume after a stop requires the explicit `--clear-stop`
flag. The wall-time budget similarly stops running managed jobs and blocks undispatched work. A
runner that supports checkpointing or semantic resume is responsible for using it on the next
explicit resume; unrelated processes are never targeted.

## Delegation boundary

Initially use no more than two implementation workers and one substantial training/evaluation job
at once; this two-core VM usually warrants lower concurrency. Every coding worker gets a separate
Git worktree, a bounded prompt, acceptance criteria, and a completion report. One lead integrates
shared code/status changes. Successive generations remain dependent tasks, never parallel jobs.

A fresh-context reviewer receives the agreement, protocol, relevant source, and actual artifacts.
It must challenge interpretation and list alternative explanations. Its judgment is an additional
check, not scientific proof.

## Durability matrix

| Event | Current support | Evidence/limitation |
| --- | --- | --- |
| Browser or SSH disconnect | Supported when launched in detached `tmux` | Managed command does not depend on the browser/SSH process |
| Lead conversation turn ends | Supported | Detached command continues; Shelley CLI can send one completion message |
| Supervisor process exits | Supported for resume | Atomic state, per-attempt wrapper records, output validation, and idempotent commands |
| Shelley service restarts | Jobs continue; notification may need resume | Job process is separate; no end-to-end service-restart test was run during setup |
| VM reboot | Artifacts survive; automatic execution resume is **not configured** | No enabled project systemd unit was installed; a lead must relaunch `resume` |
| VM loss/disk loss | Not solved by this coordinator | Claim artifacts still require the protocol's checksum archive and durable secondary copy |

The setup tests simulate supervisor interruption and exercise real child processes. They do not
claim proof of browser disconnect, Shelley restart, or VM reboot. Those boundaries remain as stated
above.

## Cycle finish

The workflow finishes when approved deliverables complete, the wall-time budget is exhausted, a
scientific-validity dependency fails, the user stops it, or meaningful progress is blocked. The
briefing and `docs/STATUS.md` update are the final authorized outputs. The coordinator then waits;
it does not invent or launch another cycle.
