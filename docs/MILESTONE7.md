# Milestone 7: Controlled RL Environment and Recipe Experiments

**Status:** Planned; not started
**Date:** September 2, 2026
**Detailed experiment list:** [Neural AI plan — Milestone 7](NEURAL_AI_PLAN.md#milestone-7-experiments-only-after-the-mc-baseline)

## Goal

Turn the completed deterministic engine and frozen-generation baseline into a systematic reinforcement
learning laboratory. Milestone 7 adds standard environment adapters and a generic recipe/experiment
layer, then uses them to compare encoder, target, model, exploration, replay, and search ideas without
losing the permanent q0/Milestone 6 baseline.

The objective is not “try increasingly complicated neural networks.” It is to identify which
subsystem limits playing strength through controlled, reproducible comparisons.

## Prerequisites

Begin claim-generating Milestone 7 experiments only after:

- q0 is preserved as the permanent first learned baseline;
- at least one full Milestone 6 generation has run end to end;
- the promotion and artifact workflow has demonstrated reproducibility at production scale;
- development and locked-final evaluation seed domains are clearly separated; and
- the current baseline's strength, throughput, coverage, and calibration diagnostics are recorded.

Environment-adapter design may begin earlier, but it must not delay or redefine the frozen Milestone
6 baseline.

## Workstream 1: RL environment adapters

The engine remains framework-independent. Add adapters in a separate optional package boundary:

```text
engine state machine
    -> safe PlayerObservation + semantic legal actions
    -> turn-based environment adapter
    -> Gymnasium/PettingZoo API and versioned tensor encoder
```

### Common turn-based adapter

Define one internal environment contract that:

- owns a seeded authoritative game state;
- exposes only the current actor's safe observation and public decision context;
- maps a stable versioned action vocabulary to semantic engine actions;
- supplies a legal-action mask;
- advances exactly one semantic player decision per multi-agent step;
- reports terminal rewards and complete reproducibility metadata; and
- supports deterministic reset/replay from declared seeds.

Do not put framework action indices into the engine. The adapter owns the reversible mapping between
a fixed external action vocabulary and stable semantic actions.

### PettingZoo adapter

A PettingZoo AEC-style interface is the natural representation of the alternating two-player game.
It should expose each player only when that player acts, preserve viewpoint-relative observations,
and support independent policies for self-play and evaluation.

### Gymnasium adapter

Provide a single-agent wrapper for common RL libraries:

- choose one learning seat;
- configure a frozen opponent policy;
- automatically advance opponent decisions between learner steps;
- return rewards from the learner's viewpoint; and
- include legal masks and opponent/checkpoint identity in reset metadata.

The two adapters must share the same semantic environment core and pass cross-adapter trajectory
equivalence tests.

### Environment acceptance criteria

- fixed seeds and semantic actions reproduce engine trajectories exactly;
- no opponent hand, deck order, or unknown face-down identity enters an observation;
- every legal semantic action is representable and every masked action maps back unambiguously;
- illegal, stale, and out-of-turn actions fail clearly;
- seat swaps canonicalize observations and rewards correctly;
- plain core installation still does not require Gymnasium, PettingZoo, NumPy, or Torch; and
- environment wrappers add negligible overhead relative to measured policy inference where
  practical.

## Workstream 2: Generic recipe and experiment layer

A recipe is broader than model shape. It fixes:

- observation/encoder version;
- external action vocabulary and mask version;
- model architecture;
- target semantics;
- behavior/self-play policy;
- corpus composition and size;
- exploration schedule;
- optimizer and training defaults;
- initialization or parent-checkpoint rule;
- evaluation and promotion gates; and
- declared training and evaluation seed domains.

Add committed machine-readable recipe declarations and a runner that can:

1. validate a recipe before work begins;
2. create isolated replicate plans;
3. resume collection/training/evaluation without changing identity;
4. aggregate multiple independent training replicates;
5. compare a candidate recipe with its declared parent on matched arenas;
6. reject incompatible artifacts rather than silently migrating them; and
7. produce a compact result document from machine artifacts.

Changing an encoder, target, or model incompatibly starts a new recipe family. Shape-compatible
experiments may warm-start only when the declaration explicitly permits it.

## Experimental method

Follow [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md). In particular:

- state a hypothesis and primary metric before running;
- change one major factor when possible;
- keep the simpler permanent baseline runnable;
- use fixed corpora for encoder/model ablations when their semantics permit it;
- use fresh self-play when the behavior policy or target requires it;
- compare on matched paired seed blocks;
- use multiple independent corpus/training replicates for architecture claims;
- reserve locked-final seeds until a recipe family is selected; and
- treat failure or practical equivalence as useful evidence.

A recipe may improve validation metrics without improving gameplay. Arena strength remains the
primary decision criterion.

## Ordered experiment program

The initial ordering from [`NEURAL_AI_PLAN.md`](NEURAL_AI_PLAN.md) remains the default. Reorder only
when measured diagnostics justify it.

### 1. Public-history encoder v2

Add fixed padded public history or compact recent-turn features. Re-encode retained semantic corpora
where valid and compare against `candidate-public-v1` using the same model/training recipe and
matched arenas.

Question: does omitted action history materially limit belief-sensitive decisions?

### 2. Blended Monte Carlo and TD(lambda)

Add sequence-level lambda returns with gamma 1, a frozen target checkpoint, and a retained terminal
Monte Carlo component.

Question: can denser temporal targets improve learning without unstable maximization bias or loss of
clear value semantics?

### 3. Opponent/champion pool

Sample frozen older checkpoints only if Milestone 6 measures cycling, forgetting, or narrow
best-response behavior.

Question: does policy diversity improve robustness relative to frozen incumbent-only self-play?

### 4. Heuristic-ranking warm start

Pretrain safe candidate rankings from `greedy-public-v1`, then fine-tune on terminal outcomes.

Question: is weak chosen-action coverage preventing useful early value estimates?

### 5. Search distillation

Use information-safe shallow rollouts or a learned belief model to produce candidate rankings, then
distill them into the direct policy/value model.

Question: can limited planning improve tactical decisions while retaining fast deployment?

### 6. Larger or deeper MLP

Increase model capacity only after encoder/data diagnostics show underfitting.

Question: is representation capacity, rather than data or targets, the measured bottleneck?

### 7. Policy head

Add a policy head only if repeated candidate evaluation or planned search makes it useful.

Question: does direct action prediction improve throughput, exploration, or search guidance without
reducing strength?

### 8. Belief-aware tree search

Treat ReBeL-style or other hidden-information search as a separate recipe family after direct value
learning and simpler history/belief experiments plateau.

Question: does explicit belief-conditioned planning justify its complexity and compute cost?

## Standard comparison report

Each Milestone 7 result should include:

- recipe and parent identities;
- hypothesis and changed factor;
- corpus count, decision count, action/phase/card coverage, and artifact size;
- per-replicate and aggregate training metrics;
- per-replicate paired arena results against the parent and guardrail baselines;
- seat splits, score margin, terminal reasons, game length, and throughput;
- bootstrap uncertainty over paired gameplay blocks and variation across training replicates;
- hidden-information and replay-equivalence checks;
- locked-final results only when the recipe was selected before opening that block; and
- an explicit conclusion: improve, reject, equivalent, or inconclusive.

## Definition of done

Milestone 7's infrastructure phase is complete when:

- a shared turn-based environment core has tested Gymnasium and PettingZoo adapters;
- environment trajectories are semantically equivalent to direct engine trajectories;
- a versioned fixed action vocabulary and legal masks are frozen;
- machine-readable recipes can run isolated reproducible replicates;
- result aggregation separates gameplay-block uncertainty from training-replicate variation;
- q0 and the Milestone 6 baseline remain runnable without migration; and
- at least one encoder/model/target experiment is completed and reported under the protocol.

The broader research milestone remains open while measured experiments continue.

## Out of scope

Milestone 7 does not imply:

- public hosting of the game;
- support for untrusted in-process agents;
- mandatory distributed/GPU training;
- unlimited architecture search;
- claiming optimal or equilibrium play from self-play results; or
- replacing semantic engine actions with framework-specific action indices.
