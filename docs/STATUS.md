# Project status

**As of:** September 2, 2026
**Current learned incumbent:** q0
**Latest completed learned result:** [Milestone 5 q0 evaluation](MILESTONE5_RESULTS.md)

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
| [6: frozen iterative self-play](MILESTONE6.md) | Core orchestration complete; production/archive work pending | Resumable `iterate` command and tiny end-to-end test |
| [7: controlled RL experiments](MILESTONE7.md) | Planned; not started | Ordered experiment plan only |

## Current measured policies

| Policy | Training source | Direct measured result |
| --- | --- | --- |
| `random-v1` | None | Symmetry/control baseline |
| `greedy-public-v1` | Hand-authored public heuristic | 86.25% vs random over 400 games |
| q0 | 4,000 epsilon-heuristic games, 62,184 decisions | 75.375% vs random and 60.125% vs heuristic over 800 games each |

The heuristic-versus-random and q0-versus-random arenas used different declared seed sets, so their
percentages should not be treated as a transitive ranking. The direct q0-versus-heuristic arena is
the relevant comparison and favored q0 with a paired-bootstrap 95% interval of 56.75%–63.50%.

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

The scheduling, restart-safe storage, training, arena, bootstrap, confirmation, and immutable
decision machinery are implemented. Only tiny test iterations have run. Before a claim-generating
q1 run, the iteration artifacts should also record the exact Git/lockfile revision and retain
compressed individual arena game records under the new experiment protocol. There is no production
q1 checkpoint or Milestone 6 result yet, so q0 remains the incumbent.

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

The q0 source corpus, dataset, checkpoint, logs, and reports are retained as an ignored archive with
an external copy. The q0 arena aggregate reports are retained, but their individual semantic game
records predate the current retention policy and were not written. Generated artifacts stay out of
ordinary Git history; compact results and exact source revisions remain committed. See
[the experiment protocol](EXPERIMENT_PROTOCOL.md) for the retention and reporting contract.

## Verified source revisions

| Purpose | Revision | Package code fingerprint |
| --- | --- | --- |
| q0 corpus, dataset, and training | `7fe2e5af6339efc86db285349043bb16b298eed1` | `b92b07727d89eb2f9471316fa44992a2eb337da2186a25f3bc4637404bb09d8d` |
| q0 learned-agent arena evaluation | `d46350356a7bf461468ba33c6bf1f28b3231cd14` | `6909e9316510f48cad515b99a8f9b19288ee5c43d5925f84b88ad853479031bc` |
| Milestone 6 iteration orchestration | `de5bb2a27f93b0d5fa31540a0dc6cec843c9fb7c` | `a679bc750f6c976e5ed39dec3166b9859a76b8141c50b3107d09feeea411950f` |

## Planned next work

These items are intentionally not complete yet:

1. add q0/current-champion selection to the web QA UI;
2. run the first full q1 Milestone 6 iteration;
3. record its promotion or retention result; and
4. after the frozen-generation baseline is established, build the generic recipe and
   Gymnasium/PettingZoo experiment layer described in [Milestone 7](MILESTONE7.md).
