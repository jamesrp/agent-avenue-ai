# Project status

**As of:** September 3, 2026
**Current learned incumbent:** q0
**Latest completed learned result:** [Milestone 6 frozen self-play](MILESTONE6_RESULTS.md)

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
| [7: controlled RL experiments](MILESTONE7.md) | Planned; not started | Ordered experiment plan only |

## Current measured policies

| Policy | Training source | Direct measured result |
| --- | --- | --- |
| `random-v1` | None | Symmetry/control baseline |
| `greedy-public-v1` | Hand-authored public heuristic | 86.25% vs random over 400 games |
| q0 | 4,000 epsilon-heuristic games, 62,184 decisions | Locked final: 75.5% vs random and 57.3% vs heuristic over 1,000 games each |

The locked final q0 intervals are 72.9%–78.1% against random and 54.2%–60.4% against the heuristic.
The heuristic-versus-random baseline used a different seed set, so percentages should not be treated
as a transitive ranking.

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

q0 is not a policy-gradient agent, a search agent, or evidence of equilibrium play. It is the first
frozen learned baseline and current incumbent for later self-play.

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
- paired, seat-swapped arena blocks; and
- immutable promotion decisions.

The q0 source corpus, dataset, checkpoint, logs, and reports remain retained. The q0 historical arena
aggregates predate the individual-record policy. Every Milestone 6 training and evaluation arena
retains compressed semantic records, and the complete run is packaged in a verified ignored archive
with an embedded checksum manifest. Generated artifacts stay out of ordinary Git history; compact
results and exact source revisions remain committed. See [the result](MILESTONE6_RESULTS.md) and
[the experiment protocol](EXPERIMENT_PROTOCOL.md).

## Verified source revisions

| Purpose | Revision | Package code fingerprint |
| --- | --- | --- |
| q0 corpus, dataset, and training | `7fe2e5af6339efc86db285349043bb16b298eed1` | `b92b07727d89eb2f9471316fa44992a2eb337da2186a25f3bc4637404bb09d8d` |
| q0 learned-agent arena evaluation | `d46350356a7bf461468ba33c6bf1f28b3231cd14` | `6909e9316510f48cad515b99a8f9b19288ee5c43d5925f84b88ad853479031bc` |
| Milestone 6 production q1–q4 and final evaluation | `139318bad909438e8a3e1cb9dd962c80653875d0` | `5a66a4a63f7c5680f3db81e83dfebf53af63f73d29b653645850bb013a45ba0c` |

## Planned next work

1. run the implemented [`terminal-safety-v1` hybrid-policy experiment](TERMINAL_SAFETY_EXPERIMENT.md),
   retraining a separately named q0 snapshot and repeating q1–q4 without rewriting historical
   artifacts;
2. add q0/current-champion selection to the optional web QA UI; and
3. begin the controlled replay, target, and environment-adapter experiments described in
   [Milestone 7](MILESTONE7.md).
