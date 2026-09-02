# Experiment protocol

**Status:** Active project policy
**Date:** September 2, 2026

## Purpose

The project should improve through controlled, reproducible experiments rather than a sequence of
unrecorded parameter changes. Every result must answer a declared question, preserve enough state to
reproduce or audit it, and distinguish predictive validation metrics from actual playing strength.

The default mindset is systems debugging:

1. define the expected behavior;
2. isolate one important change;
3. measure the whole pipeline;
4. retain the inputs and identities needed to reconstruct the result;
5. investigate failures before adding complexity; and
6. report negative or inconclusive outcomes without moving the goalposts.

## Experiment lifecycle

### 1. Declare

Before a claim-generating run, record:

- experiment ID and human-readable hypothesis;
- parent experiment, incumbent, or permanent baseline;
- the one major factor intended to change;
- exact Git revision and `uv.lock` revision;
- rules, observation, encoder, model, target, agent, RNG, and runner versions;
- normalized corpus, training, and evaluation configuration;
- all root seeds and seed-derivation domains;
- game/sample budget;
- development and locked-final evaluation blocks;
- number of independent corpus/training replicates;
- primary metric, guardrails, and success criterion; and
- expected artifact paths and retention class.

A generated `plan.json` may freeze the executable declaration, but the important recipe and result
summary must also remain discoverable from committed documentation. Future generic experiment
orchestration should use committed machine-readable recipes rather than requiring command history as
the only declaration.

### 2. Collect

Generate games through trusted runners using explicit setup and agent RNG streams. Retain complete
semantic `GameRecord`s in a compressed corpus. Verify every record before accepting it into a corpus
or extracting model samples.

Training corpora are omniscient research artifacts: setup seed plus semantic actions can reconstruct
hidden deck order, hands, and face-down cards. They must never be exposed to live agents or the web
browser. The model-input safety boundary remains `PlayerObservation + Action`.

### 3. Build and train

Materialize versioned datasets from verified records. Split by complete game or paired block, never
by individual decision. Record sample provenance separately from model inputs.

Training must record:

- data and split identities;
- initialization, shuffle, optimizer, and thread settings;
- epoch history and selected checkpoint epoch;
- train/validation loss, Brier score, calibration, accuracy, and target balance;
- phase and action/card coverage;
- examples per second and wall-clock duration; and
- deterministic tensor digest and checkpoint identity.

A lower validation loss is diagnostic evidence, not a strength claim.

### 4. Evaluate

Gameplay evaluation uses fresh setup seeds not present in training. Every matchup alternates seats
within each setup-seed block and preserves the logical agents' RNG identities across the seat swap.
The paired setup-seed block, not an individual game, is the default independent observation unit.

Required gameplay metrics are:

- paired win rate and deterministic block-bootstrap confidence interval;
- ordinary Wilson interval for continuity;
- win rate and game count by seat;
- average score margin;
- average turns and decisions;
- terminal-reason counts;
- games per second and candidate evaluations per second where available; and
- complete agent, checkpoint, rules, code, configuration, and seed identities.

Useful diagnostics that should be added when available include action/card frequency, top-two
candidate score margin, calibration by decision phase, and performance by public state bucket. Such
diagnostics must use only safe observations or trusted offline records and must not alter the
predeclared promotion statistic.

### 5. Decide

Promotion and architecture-selection rules are fixed before seeing the result. A failed or
inconclusive run retains the incumbent unless the declared protocol says otherwise. Do not add seed
blocks, change confidence levels, weaken guardrails, or select a more favorable metric after seeing
the initial result.

The Milestone 6 gate is defined in [the neural plan](NEURAL_AI_PLAN.md#promotion-gate) and summarized
in [the Milestone 6 reference](MILESTONE6.md). An immutable decision artifact records the selected
checkpoint and all evidence used.

### 6. Record

Every completed claim-generating experiment gets a compact committed result note containing:

- question and declared change;
- status: success, failure, inconclusive, or infrastructure-only;
- exact source revisions;
- artifact and checkpoint fingerprints;
- data/training/evaluation seed roots;
- primary and guardrail metrics with uncertainty;
- important diagnostics and throughput;
- deviations from the declaration, if any;
- interpretation limited to what was measured; and
- next action justified by the evidence.

`docs/STATUS.md` links the current incumbent and latest important results. Detailed historical reports
remain immutable except for explicit corrections or added archival metadata.

## Evaluation blocks and overfitting control

Use three classes of evaluation:

1. **Smoke blocks:** tiny, fast checks for wiring and regressions. Never support strength claims.
2. **Development/promotion blocks:** predeclared paired seeds used for routine comparisons and
   promotion decisions.
3. **Locked final blocks:** untouched seeds used only after selecting a recipe or final champion.

Once results from a seed block inform model or recipe selection, that block is development data even
if it was originally called a test. Do not repeatedly tune against q0's historical arena seeds and
later describe those seeds as locked final evaluation.

Use matched setup-seed blocks when comparing two agents or recipes. For architecture claims, report
multiple independent corpus/training replicates when the budget permits. A single replicate is fine
for smoke debugging and the frozen baseline chain, but it does not measure sensitivity to training
randomness.

Recommended default for Milestone 7 recipe claims:

- one smoke replicate while debugging;
- at least three independent declared training replicates for the selected comparison; and
- matched development arenas for each replicate, followed by one locked-final evaluation only after
  selecting the recipe family.

## Change isolation

Prefer one major changed factor per experiment:

- encoder representation;
- target construction;
- model architecture;
- exploration policy;
- corpus composition;
- optimizer/training recipe; or
- inference/search policy.

Supporting implementation changes may be necessary, but they must not silently change another
semantic factor. If multiple factors must change together, name the experiment as a new recipe
family and do not attribute its result to one component.

Permanent baselines remain runnable. New approaches do not overwrite or redefine the original q0
Monte Carlo baseline.

## Artifact retention

### Tier 1: committed, compact records

Keep in Git:

- source code and `uv.lock`;
- milestone and result documents;
- small benchmark specifications and fixtures;
- exact source revisions and fingerprints;
- normalized experiment recipes when the recipe format is implemented; and
- compact aggregate result summaries.

Do not commit generated corpora, datasets, checkpoints, large reports, or replay collections to the
normal repository history.

### Tier 2: retained research artifacts

Retain outside Git, with a checksum manifest and a durable external copy, for all future
claim-generating runs:

- complete compressed semantic training corpora;
- materialized datasets;
- every promoted checkpoint;
- rejected candidate checkpoints used in a reported comparison;
- arena aggregate reports;
- compressed individual arena game records; and
- iteration plans and promotion decisions.

Measured q0 storage is approximately:

| Artifact group | Size |
| --- | ---: |
| 4,000-game compressed corpus and manifest | 1.5 MB |
| Dataset and manifest | 2.0 MB |
| Checkpoint bundle | 0.4 MB |
| Training/evaluation logs and reports | 0.8 MB |
| Total | about 4.7 MB |

At this scale, retaining O(number of games) semantic records is preferable to discarding them. The
planned five-generation baseline should remain in the tens of megabytes; even conservative overhead
leaves ample room for complete records.

The q0 arenas predate this policy and did not retain individual evaluation game records. That
exception is documented in [`MILESTONE5_RESULTS.md`](MILESTONE5_RESULTS.md); future claim-generating
arenas retain them.

### Tier 3: disposable derived outputs

Plots, reformatted tables, caches, temporary shards after corpus finalization, and debug traces may
be regenerated from Tier 1 and Tier 2 artifacts. Retain them only when they explain a published
result or an otherwise hard-to-reconstruct failure.

## Reproduction contract

A result is reproducible when the declared source revision, lockfile, environment, root seeds,
normalized recipe, and artifact inputs produce the same semantic schedules, game-record
fingerprints, dataset identity, model tensor digest, and gameplay statistics. Timestamped JSON,
compressed bytes, wall-clock timing, and filesystem paths need not be byte-identical unless a format
explicitly declares that requirement.

Because deterministic neural behavior can depend on library and platform details, retaining the
actual checkpoint and corpus is stronger than relying only on regeneration. Source recipes and
seeds are still mandatory: an opaque checkpoint without lineage is not an acceptable archived
result.

## Debugging checklist

When an experiment surprises us, inspect in this order:

1. compatibility, schema, digest, and replay verification;
2. train/evaluation seed overlap and seat balance;
3. hidden-information invariance;
4. target actor perspective and terminal labeling;
5. corpus action/phase/card coverage;
6. train/validation loss, calibration, and target balance;
7. action distributions and candidate score margins;
8. performance split by seat, phase, terminal reason, and public state bucket;
9. deterministic rerun on a small fixed fixture; and
10. only then model capacity, optimizer changes, TD targets, opponent pools, or search.

The default response to a weak model is to locate the limiting subsystem, not immediately make the
network larger.
