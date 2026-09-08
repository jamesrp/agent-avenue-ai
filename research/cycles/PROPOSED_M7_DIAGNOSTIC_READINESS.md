# Proposed first research-cycle agreement: Milestone 7 diagnostic readiness

**Cycle ID:** `m7-diagnostic-readiness-v1`  
**Status:** Proposed; awaiting user approval  
**Proposed:** September 8, 2026  
**Parent evidence:** terminal-safety-v1 completed result

## Research question

Given the completed q1–q4 and terminal-safety results, what is the best-supported **first** Milestone 7
data/target experiment to run next, and can the selected historical/current policies be made directly
inspectable in web QA without weakening the information boundary?

This matters because another incumbent-only q1 iteration is already answered and would repeat
completed work. The next expensive experiment should target a measured failure mode rather than add
model complexity by intuition.

## Hypotheses

1. The retained q1–q4 heuristic regressions and all-pairs prediction-transfer diagnostics will be
   more consistent with narrow selected-action/opponent coverage than with simple model undercapacity.
2. Deterministic retrospective diagnostics can narrow the next choice to one of mixed-opponent replay,
   counterfactual candidate-ranking supervision, or public-history encoder v2, but cannot establish
   causal playing-strength improvement without a later controlled training experiment.
3. A server-side allowlist for historical q0 and `q0-terminal-safety-v1` can provide hand-checkable
   behavior examples while preserving `PlayerObservation + Action` as the only model input.

## Work packages

### A. Retention and recovery audit (prerequisite)

- Inventory the exact live and archived q0/Milestone 6/terminal-safety artifacts.
- Verify checksums and perform a disposable extraction/restore validation of both major archives.
- Confirm the workflow and existing resumable corpus/iteration recovery tests at the cycle source.
- Produce `artifact-catalog.json` and a short retention-gap note.

**Complete when:** every artifact used by B has a verified locator/fingerprint; missing or corrupt
inputs block only dependent diagnostics. Off-VM durability is reported honestly, not implied.

### B. Deterministic diagnostic evidence pack (depends on A)

- Recompute key retained aggregates from individual records where available.
- Summarize action/card/phase coverage, target balance, calibration/transfer evidence, seat effects,
  terminal reasons, and the q1–q4 heuristic-guardrail pattern.
- Include a few hand-checkable replay positions showing the acting player's visible information and
  candidate choices; do not expose hidden state to live-agent or browser code.
- Compare the evidence for mixed-opponent replay, candidate-ranking supervision, history features,
  and larger capacity; recommend one next controlled experiment without running it.

**Complete when:** deterministic tables/plots and trace examples link to source artifacts, every
number is reproducible by a command, and conclusions distinguish evidence from alternative
explanations.

### C. Learned-checkpoint web QA (parallel with B after A identifies checkpoints)

- Add a server-configured, server-side allowlist for historical q0 and the current hybrid champion.
- Validate checkpoint compatibility before game creation; never accept a browser path.
- Show the selected label and immutable fingerprint; support either human seat and replay/start-over.
- Extend automated HTTP/security tests and the manual QA checklist.

**Complete when:** tests prove no arbitrary path/private model state reaches the browser, both
allowlisted policies can play from either seat, and ordinary non-RL core tests remain usable.

### D. Independent audit and briefing (depends on A, B, and C)

- Give a fresh-context reviewer this agreement, the experiment protocol, relevant code, raw
  artifacts, deterministic analysis, and representative traces.
- Require challenges to evaluation validity, hidden-information safety, causal interpretation, and
  the recommended next experiment.
- Integrate corrections, update `docs/STATUS.md`, and prepare one coherent research briefing.

**Complete when:** the briefing states belief changes, results/uncertainty, failures/deviations,
limitations/alternatives, one board-game ML lesson, and explicit next decisions/tradeoffs.

## Frozen scientific criteria

- This is a **retrospective diagnostic/infrastructure cycle**, not a strength or promotion claim.
- No new training corpus, model training, q-generation, promotion decision, or locked-final seed is
  opened.
- Existing record-level results must reproduce their published aggregates within deterministic
  arithmetic/format tolerances; mismatches stop dependent interpretation.
- Evaluation remains paired and seat-aware. Arena sampling uncertainty is not presented as
  independent-training variability.
- Information safety remains `PlayerObservation + Action`; offline omniscient records stay out of
  agents and browser responses.
- Web QA anecdotes illustrate behavior but do not establish aggregate strength.
- Success may be a negative result: “existing evidence cannot distinguish the next intervention” is
  acceptable if supported.

## Limits

- Wall-clock operating budget: **8 hours** after dispatch; finish sooner when deliverables complete.
- VM: current CPU-only two-core machine; no GPU, infrastructure provisioning, or paid service.
- Concurrency: at most **2** total implementation/analysis jobs and at most **1** substantial
  evaluation/replay job; successive dependent stages remain sequential.
- Agents: Shelley lead, at most **2** bounded implementation workers in isolated worktrees, and
  **1** fresh-context reviewer. No LLM polling for process completion.
- Automatic retries: at most **1** repair-and-retry per failed operational step.
- Optional confirmation: only if aggregate/record validation is ambiguous, at most **100 paired
  development-seed games** or an equivalent under-30-minute deterministic replay check. It remains
  diagnostic and cannot change historical promotion decisions.
- Storage: retain compact committed reports; generated evidence under `runs/research-cycles/` and
  checksum archives under existing ignored artifact locations.

## Preapproved conditional behavior

The lead may fix ordinary implementation defects, resume compatible interrupted jobs, retry once,
skip a blocked package while continuing independent authorized work, and run the bounded confirmation
above. A scientific-validity failure blocks dependent claims and is reported rather than patched
around.

## Decisions reserved for the user

User approval is required before:

- choosing and executing the next algorithm/data/target family;
- any model training or claim-generating strength comparison;
- changing rules, observations, information boundaries, promotion criteria, or locked-final seeds;
- materially increasing the budget/concurrency; or
- adding paid/external infrastructure or publishing/deploying the game.

## Expected critical path and outputs

```text
A retention/recovery audit
├── B diagnostic evidence pack ─┐
└── C learned web QA ───────────┼── D independent audit + briefing ── await user
                                ┘
```

Expected outputs: artifact catalog, restore/recovery evidence, deterministic diagnostic tables/plots,
representative safe traces, tested web checkpoint registry, reviewer report, updated project status,
and `BRIEFING.md`. No next cycle launches automatically.
