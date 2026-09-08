# Project status

**As of:** September 8, 2026
**Current selected hybrid champion:** q0-terminal-safety-v1
**Permanent pure-neural baseline:** historical q0
**Latest completed learned result:** [Terminal-safety hybrid rerun](TERMINAL_SAFETY_RESULTS.md)

This document is the concise, living index of completed work and current next steps. Historical
milestone documents describe the intended scope at the time; `README.md` and this file describe the
actual present state.

## Milestone status

| Milestone | Status | Evidence |
| --- | --- | --- |
| [1: deterministic rules engine](MILESTONE1.md) | Complete | Engine, observation boundary, replay fixtures, and rule tests |
| [2: lightweight web QA](MILESTONE2.md) | Complete | Hot-seat UI, private views, seeded games, and HTTP tests |
| [3: baseline agents and arena](MILESTONE3.md) | Complete | [Random/heuristic baseline results](MILESTONE3_RESULTS.md) |
| [4: neural foundation](NEURAL_AI_PLAN.md#milestone-4-safe-encoder-dataset-model-and-checkpoint) | Complete | Encoder, dataset, trainer, checkpoint, and compatibility tests |
| [5: first learned checkpoint](NEURAL_AI_PLAN.md#milestone-5-learned-agent-and-first-usable-checkpoint) | Complete | [q0 result and reproduction contract](MILESTONE5_RESULTS.md) |
| [6: frozen iterative self-play](MILESTONE6.md) | Complete | [q1–q4 decisions, final evaluation, and archive](MILESTONE6_RESULTS.md) |
| [Terminal-safety hybrid](TERMINAL_SAFETY_EXPERIMENT.md) | Complete | [Shielded q0–q4 result, all-pairs diagnostics, and archive](TERMINAL_SAFETY_RESULTS.md) |
| [7: controlled RL experiments](MILESTONE7.md) | Planned; not started | Ordered experiment plan only |

## Current measured policies

| Policy | Training source | Direct measured result |
| --- | --- | --- |
| `random-v1` | None | Symmetry/control baseline |
| `greedy-public-v1` | Hand-authored public heuristic | 86.25% vs random over 400 games |
| historical q0 | 4,000 epsilon-heuristic games, 62,184 decisions | Locked final: 75.5% vs random and 57.3% vs heuristic over 1,000 games each |
| q0-terminal-safety-v1 | 4,000 shielded epsilon-heuristic games, 65,746 decisions | Locked final: 77.9% vs random, 67.4% vs heuristic, and 56.7% vs historical q0 |

The hybrid q0 paired-bootstrap intervals are 75.3%–80.5% against random, 64.4%–70.4% against the
heuristic, and 54.1%–59.3% against historical q0. Historical and hybrid final blocks use different
fresh seeds, so the improvement over the old aggregate is descriptive; the direct hybrid-versus-
historical-q0 arena is the clean head-to-head comparison.

Validation loss and accuracy measure outcome prediction on held-out games, not playing strength.
Promotion and strength claims come from paired gameplay arenas.

## What q0 is

q0 is an action-conditioned Monte Carlo value model:

```text
Q(current acting-player observation, legal semantic action)
    = probability that the acting player eventually wins
```

It uses the information-safe 87-feature `candidate-public-v1` encoder and the 11,393-parameter
`candidate-mlp-v1` network. It was trained on the action actually selected at each recorded decision,
with the final winner as the target from that decision actor's viewpoint.

Historical q0 remains the permanent pure-neural baseline rather than being overwritten by later
experiments. q0-terminal-safety-v1 uses the same action-conditioned model family and adds the narrow
public immediate-loss veto described in the terminal-safety declaration.

## What Milestone 6 means

Milestone 6 is a project phase, not “generation six.” It applies the same declared policy-improvement
loop to generations q1 through q4:

1. freeze the current incumbent;
2. collect 4,000 epsilon-incumbent self-play games;
3. train one warm-started candidate on that generation;
4. compare candidate and incumbent on fresh paired games;
5. enforce random, heuristic, seat, compatibility, and reproducibility guardrails; and
6. either promote the candidate or explicitly retain the incumbent.

The full q1–q4 chain ran on September 2, 2026. Each proposal beat q0 directly but failed the
predeclared heuristic non-regression guardrail, so every immutable decision retained q0. The chain
ended `budget_exhausted_inconclusive`; it did not meet the practical-equivalence plateau criterion.
A separate q1 regeneration reproduced all semantic identities, and q0 completed a fresh locked final
evaluation. See [the Milestone 6 result](MILESTONE6_RESULTS.md).

The terminal-safety rerun completed on September 3, 2026. The separately trained shielded q0 passed
its gate and beat historical q0 directly. All shielded q1–q4 proposals again dominated their parent
but failed the aligned heuristic non-regression guardrail, so shielded q0 was retained under the
predeclared gate. Across 591,219 audited shielded decisions, the safety invariant held with zero
executed avoidable provable losses; this is retained-record coverage rather than a universal proof.

The additional held-out diagnostic evaluated each `qn` on fresh records from every unordered pair in
`(heuristic, q0, ..., q(n-1))`. It covered 14,000 games and exposed poor prediction transfer to
interactions among rejected candidates. Those records were diagnostic-only and did not change
fitting or promotion. See [the terminal-safety result](TERMINAL_SAFETY_RESULTS.md).

## Web status

The QA web UI currently supports:

- human versus human;
- human versus random;
- human versus `greedy-public-v1`; and
- either human seat for automated modes.

Learned-checkpoint play is not implemented. The intended next UI extension is a server-side registry
of validated checkpoints such as q0 and the current champion. The browser must not submit arbitrary
checkpoint paths or receive private model/environment state.

## Reproducibility status

The repository currently records:

- normalized game, agent, encoder, model, training, and iteration configurations;
- domain-separated setup, agent, split, training, arena, and bootstrap seeds;
- rules, code, corpus, dataset, model tensor, checkpoint, and report fingerprints;
- semantic actions in retained corpora and game records sufficient to replay those completed games;
- deterministic game-level data splits;
- paired, seat-swapped arena blocks;
- immutable promotion decisions;
- replay-derived immediate-loss safety diagnostics; and
- held-out, setup-disjoint all-pairs prior-policy evaluation plans and reports.

The q0 source corpus, dataset, checkpoint, logs, and reports remain retained. The q0 historical arena
aggregates predate the individual-record policy. Every Milestone 6 and terminal-safety training and
evaluation arena retains compressed semantic records. Both completed chains are packaged in verified
ignored archives with embedded member checksums. Generated artifacts stay out of ordinary Git
history; compact results and exact source revisions remain committed. See the
[terminal-safety result](TERMINAL_SAFETY_RESULTS.md), [Milestone 6 result](MILESTONE6_RESULTS.md),
and [experiment protocol](EXPERIMENT_PROTOCOL.md).

## Verified source revisions

| Purpose | Revision | Package code fingerprint |
| --- | --- | --- |
| q0 corpus, dataset, and training | `7fe2e5af6339efc86db285349043bb16b298eed1` | `b92b07727d89eb2f9471316fa44992a2eb337da2186a25f3bc4637404bb09d8d` |
| q0 learned-agent arena evaluation | `d46350356a7bf461468ba33c6bf1f28b3231cd14` | `6909e9316510f48cad515b99a8f9b19288ee5c43d5925f84b88ad853479031bc` |
| Milestone 6 production q1–q4 and final evaluation | `139318bad909438e8a3e1cb9dd962c80653875d0` | `5a66a4a63f7c5680f3db81e83dfebf53af63f73d29b653645850bb013a45ba0c` |
| Terminal-safety q0–q4, all-pairs diagnostics, and final | `19c2871080503c62a53522415e0645913d7674b0` | `9c8b39bb4e81bb29e0902c56198557f63b7e9992bdaff845967c562fc064f429` |

## Research workflow status

A bounded coordinator and cycle-record convention are implemented and tested; see
[`RESEARCH_WORKFLOW.md`](RESEARCH_WORKFLOW.md) and
[`RESEARCH_WORKFLOW_INVENTORY.md`](RESEARCH_WORKFLOW_INVENTORY.md). It records queued/running/
completed/failed/blocked/stopped work, exact attempts and exit status, output evidence, bounded
retries, source freezing, stop/budget state, and one optional Shelley completion message. It wraps
rather than duplicates the existing experiment runners.

The setup-only declarations are `research/cycles/setup-smoke-v1.json` through v3. V1 retained the
controlled schema-validation failure and bounded retry; v2 demonstrated detached automatic stage
progression and Shelley continuation; review-hardened v3 completed with source snapshots and output
digests. Generated evidence lives under ignored `runs/research-cycles/setup-smoke-v*/` and does not
support a gameplay-strength claim. See [`RESEARCH_WORKFLOW_SETUP_REPORT.md`](RESEARCH_WORKFLOW_SETUP_REPORT.md).
No real research cycle is approved or running. The proposed first agreement is
[`research/cycles/PROPOSED_M7_DIAGNOSTIC_READINESS.md`](../research/cycles/PROPOSED_M7_DIAGNOSTIC_READINESS.md).

A real retrospective cycle is now approved and in progress:
[`m7-diagnostic-readiness-v1`](../research/cycles/M7_DIAGNOSTIC_READINESS_V1.md). Its frozen executable
plan is `research/cycles/m7-diagnostic-readiness-v1.json`. It permits retention/restore validation,
retained-record diagnostics, learned-checkpoint web QA, independent review, and one briefing; it
forbids training, promotion, locked-final evaluation, and automatic next-cycle launch.

Verified setup durability is limited to local files/processes and explicit resume. VM-reboot
execution resume and off-VM artifact durability are not configured or claimed.


1. predeclare the next data/target experiment, using the terminal-safety hybrid champion while
   testing mixed-opponent replay and/or counterfactual candidate-ranking supervision rather than
   extending incumbent-only selected-action Monte Carlo training;
2. add historical q0/current hybrid champion selection to the optional web QA UI; and
3. implement the recipe and environment-adapter boundaries described in [Milestone 7](MILESTONE7.md).
