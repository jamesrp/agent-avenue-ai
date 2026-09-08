# Bounded research workflow setup report

**Date:** September 8, 2026  
**Status:** Setup implemented and smoke-tested; no real research cycle started  
**Latest coordinator correction:** `3ce3d49` (`Serialize stop control without state races`)

## Executive result

The project now has a small, repository-specific coordinator around its existing experiment runners.
It can execute a fingerprinted approved task graph, persist task/attempt state, run ordinary managed
processes, validate output evidence, resume without rerunning completed work, enforce one substantial
job and bounded retries, stop safely, end on budget, send one Shelley continuation message, and stop
without inventing a next cycle.

This is not a claim that the VM is a fault-tolerant service or that the coordinator validates game
science by itself. Claim-generating work must still use the existing trusted runners and protocol.

## What changed

- Added `agent_avenue.research` with `validate`, `run`, `resume`, `status`, and `stop` commands.
- Added an OS-process wrapper that records exact command, cwd, PID, stdout/stderr, timeout/signal, and
  exit code per attempt.
- Added atomic `state.json` plus separate race-safe `control.json` for stop requests.
- Added dependency scheduling, workflow/task states, one-retry maximum, wall-time termination, and a
  one-substantial-job limit.
- Added clean-source snapshots for frozen tasks, runtime/workdir containment checks, required JSON
  keys for claim manifests, and SHA-256 evidence revalidation for completed outputs.
- Added the bounded operating contract in `docs/RESEARCH_WORKFLOW.md` and conventions in `AGENTS.md`.
- Added current capability/artifact inventory and updated `README.md`/`docs/STATUS.md`.
- Added setup-only executable plans and the proposed first real cycle agreement under
  `research/cycles/`.
- Created a byte-identical same-VM secondary copy of the terminal-safety archive and verified its
  SHA-256 plus gzip/tar listing.

## Bounded delegation evidence

The first isolated audit worker returned no artifact after one follow-up. A second and final
implementation worker used `/tmp/agent-avenue-workflow-audit-2`, committed
`f3080f8` (`Add research workflow inventory`), and reported a passing then-current full check. The
lead cherry-picked it as `cc4eba8`, rechecked ignored artifacts in the authoritative primary
checkout, and corrected the inventory where an isolated worktree could not see those artifacts.

This demonstrates bounded delegation, isolated edits, collection, integration, and lead validation.
The failed first worker is retained as a limitation rather than reported as completion.

## Miniature cycle evidence

### Controlled failure and bounded repair

`setup-smoke-v1` ran a real four-pair random-versus-random arena but expected the wrong report key
(`game_count` rather than `total_games`). The command exited zero, output validation failed, exactly
one automatic retry occurred, then the experiment failed and all dependent stages became blocked.

Evidence: `runs/research-cycles/setup-smoke-v1/state.json`.

The lead diagnosed the schema mismatch and created versioned `setup-smoke-v2` rather than rewriting
the executed v1 plan. V2 completed in detached `tmux`, automatically advancing:

```text
delegation integration check
  -> 8-game paired arena
  -> deterministic analysis
  -> independent metric recomputation
  -> smoke briefing
```

A skeptical fresh-context review rejected that snapshot and identified a real stop/state race plus
several overbroad claims. One bounded repair added output digests, path containment, stronger
claim-plan requirements, all-stage source snapshots, and a separate stop control file.
`setup-smoke-v3` then completed the same five stages once at frozen source `deae212`; every declared
output has retained SHA-256 evidence, and an explicit resume preserved one attempt per task.

Evidence:

- `runs/research-cycles/setup-smoke-v3/state.json`
- `runs/research-cycles/setup-smoke-v3/analysis/summary.json`
- `runs/research-cycles/setup-smoke-v3/review/review.json`
- `runs/research-cycles/setup-smoke-v3/BRIEFING.md`
- `runs/research-cycles/setup-smoke-v3/resume-after-completion.json`

The v3 arena was 8 disposable random-versus-random games in four seat-swapped blocks. Agent A won
5/8; the paired-bootstrap interval was 0.500–0.875. This is explicitly **smoke-only** and supports no
playing-strength claim.

### Interruption, retry, stop, and budget

The workflow test suite launches real child processes and retains disposable scenario states under
`runs/research-cycles/setup-smoke-v3/pytest-evidence/`. It covers:

- one failed attempt followed by one successful retry;
- permanent failure blocking a dependent while independent work completes;
- supervisor termination and resume without duplicating the completed command;
- stop terminating a managed job and preventing downstream dispatch;
- a stop-control write that does not race by rewriting supervisor state;
- wall-time budget termination and blocked follow-up work;
- changed completed-output digest detection; and
- output path escape rejection.

Evidence: `workflow-tests.xml`, `workflow-tests.stdout`, and `workflow-state-locators.txt` in that
runtime directory.

### Shelley continuation

A completed v2 workflow sent one message through the installed experimental
`shelley client chat -c ...` interface into disposable conversation `c2FAJUJ`. The message appeared
as a new user turn and the conversation produced a subsequent agent turn. No model polling was used.

Evidence: `runs/research-cycles/setup-smoke-v2/notification-*.json*` and the notification section of
its `state.json`.

## Fresh-context challenge

The fresh reviewer correctly rejected v2 and then v3 snapshots rather than rubber-stamping them.
The v3 re-review is at `runs/research-cycles/setup-smoke-v3/fresh-context-rereview.md`. Its most
important remaining concrete defect—`request_stop()` rewriting `state.json` concurrently—was fixed
in `3ce3d49`; stop now writes only `control.json`, and the supervisor is the sole runtime-state
writer. The new regression test verifies the stop command does not rewrite state.

Per the one-repair-and-retry setup limit, no third reviewer pass was requested. Therefore the honest
record is: the implementation lead considers the coordinator suitable for the narrowly proposed
retrospective cycle, while the last independent review did **not** endorse its earlier snapshot.
Before any claim-generating cycle, the executable plan must supply runner-specific semantic
validators, provenance manifests, and resume/idempotence evidence; the coordinator alone is not
sufficient.

## Tests actually passed

- `make check`: Ruff, strict mypy, and the full suite passed after implementation.
- Full suite at the final setup stage: **168 tests passed** (one FastAPI/Starlette deprecation
  warning; no test failures).
- Focused workflow suite: **9 tests passed**.
- Setup v3: all five tasks completed once; post-completion resume launched no duplicate attempt.
- Terminal-safety secondary archive SHA-256 matched
  `d33c3e4bc3fea107a4d7ffe93dc841d9af8d12687269b7da783f540167fb27b9` and its tar listing passed.

## Durability boundaries

Demonstrated:

- detached process continues without an active browser/SSH command;
- conversation turn can end while managed process state remains on disk;
- supervisor process interruption can be resumed;
- duplicate completed work is rejected by state plus output digest evidence;
- stop and budget terminate only recorded managed process groups;
- a completion event can trigger one Shelley conversation message.

Not demonstrated or supported:

- automatic execution resume after VM reboot (manual `resume` is required);
- survival of VM/disk loss;
- off-VM artifact backup;
- end-to-end Shelley service restart during completion notification;
- sandboxing arbitrary commands (plans are trusted committed local inputs);
- a separate append-only state journal; or
- native Codex work (installed CLI is unauthenticated).

Same-VM secondary archive copies protect against accidental path deletion, not VM loss.

## How to use this through Shelley

1. Discuss and approve one agreement in this conversation.
2. Shelley records the exact approved version, integrates bounded implementation work, and freezes a
   clean executable plan.
3. Shelley launches the plan detached with one completion message configured.
4. Ask “status”, “stop”, or “resume” in ordinary conversation; Shelley runs the corresponding local
   command and explains the durable state.
5. On completion, Shelley validates artifacts, obtains the authorized fresh-context review, writes
   one briefing, updates project status, and waits for your next decision.

No first real cycle has been launched. The proposal is
`research/cycles/PROPOSED_M7_DIAGNOSTIC_READINESS.md`.
