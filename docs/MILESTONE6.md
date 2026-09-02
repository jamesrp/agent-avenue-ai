# Milestone 6: Frozen Iterative Self-Play and Promotion

**Status:** Core orchestration implemented and tested; production archival additions and full q1–q4
runs pending
**Date:** September 2, 2026
**Detailed specification:** [Neural AI plan — Milestone 6](NEURAL_AI_PLAN.md#milestone-6-frozen-iterative-self-play-and-promotion)

## Goal

Run reproducible offline policy-improvement generations and select a learned champion using a
predeclared statistical gate. A generation produces a candidate proposal; it does not automatically
replace the incumbent.

This milestone begins with q0 as the incumbent and plans up to four self-play generations, q1 through
q4. “Milestone 6” is the project milestone name, not a model generation number.

## Prerequisites

- Milestones 1–3 provide a deterministic engine, safe observation boundary, semantic agents,
  replayable game records, and paired arenas.
- Milestone 4 provides the versioned encoder, dataset, model, trainer, and checkpoint contracts.
- Milestone 5 provides q0 and a direct paired evaluation against random and `greedy-public-v1`.
- q0's lower paired-bootstrap endpoint against random exceeds 50%, allowing iterative strength work
  to proceed.

## Frozen-generation workflow

For generation `k`, let `c{k-1}` be the current incumbent. One declared attempt performs:

1. validate and freeze the incumbent checkpoint identity and tensor digest;
2. freeze every attempt seed, path, count, epsilon, training option, and gate in `plan.json`;
3. generate 4,000 games with the incumbent controlling both seats under seeded epsilon exploration;
4. retain complete verified semantic records in an ordered compressed corpus;
5. build a deterministic game-level 90/10 train/validation dataset from this generation only;
6. initialize a candidate from the incumbent weights;
7. train offline with the fixed `candidate-public-v1`/`candidate-mlp-v1` recipe;
8. run candidate-versus-incumbent and baseline guardrail arenas;
9. apply the predeclared promotion rule, including the one allowed confirmation block when required;
10. emit an immutable decision selecting either the candidate or retained incumbent; and
11. validate and reuse every completed artifact if the same attempt is resumed.

No checkpoint is mutated while games use it. Training occurs only after corpus collection.

## Planned schedule

| Generation | Behavior checkpoint | Self-play games | Exploration epsilon |
| --- | --- | ---: | ---: |
| q1 | current incumbent, initially q0 | 4,000 | 1/10 |
| q2 | promoted incumbent after q1, or retained q0 | 4,000 | 3/40 |
| q3 | current promoted/retained incumbent | 4,000 | 1/20 |
| q4 | current promoted/retained incumbent | 4,000 | 1/40 |

A failed attempt leaves the incumbent unchanged. A retry must use a distinct attempt ID and disjoint
corpus, training, arena, and bootstrap seed domains.

The v1 baseline trains only on the current generation so the model's value target remains tied to a
named epsilon-incumbent continuation policy. Cumulative replay is a separate Milestone 7 experiment,
not a silent change to the baseline.

## Promotion evaluation

### Primary comparison

- 500 paired setup seeds;
- 1,000 candidate-versus-incumbent games, alternating seats within each pair;
- deterministic 20,000-resample block bootstrap over the paired seed blocks; and
- promotion requires the 95% lower endpoint to be strictly above 50%.

A primary lower endpoint from 49% through 50% triggers exactly one separately seeded 1,000-pair
confirmation block. No other borderline reruns are allowed.

### Guardrails

The candidate must also satisfy:

- at least 45% point-estimate win rate from each seat in the primary arena;
- a paired-bootstrap lower endpoint above 50% against random over 200 paired seeds;
- no more than a five-point statistically supported regression relative to the incumbent against
  `greedy-public-v1`, measured on aligned paired blocks; and
- all compatibility, replay, hidden-information, and deterministic reproduction checks.

The complete formulas and bootstrap order statistics are specified in
[the neural plan](NEURAL_AI_PLAN.md#promotion-gate). Wilson intervals remain in reports for
continuity, but pair-aware bootstrap intervals support the promotion claim.

## Plateau handling

A failed promotion is not automatically a plateau. The recipe is operationally plateaued only after
the same incumbent remains in place and two consecutive independently seeded proposals are
classified as practically equivalent under the predeclared familywise interval and five-point
minimum practical effect.

If the four-generation budget ends without that evidence, report `budget exhausted, evidence
inconclusive` rather than claiming the architecture is capped out.

## Artifacts

A completed attempt contains:

```text
attempt-directory/
  plan.json
  corpus/
    manifest.json
    games.jsonl.gz
  dataset.json
  dataset.npz
  candidate/
    manifest.json
    metrics.json
    weights.pt
  arenas/
    primary.json
    versus-random.json
    candidate-versus-heuristic.json
    incumbent-versus-heuristic.json
    confirmation.json              # only when required
  arena-records/                    # required production addition; compressed semantic records
  promotion-decision.json
```

The corpus writer keeps verified per-game shards while collection is incomplete, generates only
missing games after interruption, and atomically publishes the ordered compressed corpus and
manifest when complete.

Generated artifacts stay outside normal Git history but are retained according to
[`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md).

The current implementation emits aggregate arena reports but not `arena-records/`; that directory is
part of the production artifact target rather than a description of current output.

## Current implementation

The `iterate` command implements one complete declared generation:

```bash
uv run python -m agent_avenue iterate runs/g1-attempt1 \
  --incumbent checkpoints/q0 --generation 1 --attempt-id g1-attempt1 \
  --seed 2026090100 --dry-run

uv run python -m agent_avenue iterate runs/g1-attempt1 \
  --incumbent checkpoints/q0 --generation 1 --attempt-id g1-attempt1 \
  --seed 2026090100
```

Implemented and tested behavior includes:

- deterministic attempt-specific seed domains;
- incumbent replacement/tamper rejection;
- restart-safe corpus collection;
- current-generation dataset construction;
- warm-started candidate training and lineage validation;
- primary, random, and aligned heuristic arenas;
- pair-aware promotion statistics and confirmation handling;
- immutable promotion/retention decisions; and
- full reuse/validation when an iteration command is rerun.

A tiny end-to-end test exercises the workflow. As of September 2, 2026, no full 4,000-game q1 run
has been executed and no q1 result exists.

### Production-readiness additions

Before the first claim-generating q1 run:

1. include the exact Git revision and `uv.lock` digest in `plan.json` and the promotion decision;
2. retain compressed individual semantic records for every primary, guardrail, and confirmation
   arena, not only aggregate reports; and
3. include those files in the attempt archive and checksum manifest.

The current code fingerprint and normalized configurations remain useful compatibility identities,
but they do not by themselves identify the source commit or dependency lockfile for a future reader.
These are bounded archival additions; the self-play, training, and promotion algorithms are already
implemented.

## Definition of done

Milestone 6 is complete when:

- at least one full 4,000-game generation reproduces end to end;
- up to four declared generations have been run or the recipe meets its stopping rule;
- every attempt has an immutable promotion or retention decision;
- every attempt records its exact Git revision and dependency-lock digest;
- claim-generating arena records are retained in compressed semantic form;
- the selected champion has a locked final evaluation against the applicable unique set of random,
  heuristic, q0, parent, and diagnostic earlier champions;
- final reports include paired and seat-specific uncertainty, margins, terminal reasons, game
  lengths, throughput, and artifact identities; and
- negative, equivalent, or inconclusive results are reported without weakening the gate.

A promotion is not required for the milestone to be scientifically successful.

## Out of scope

- continuously updating weights during game collection;
- distributed actors or required GPU execution;
- declaring equilibrium or optimal play;
- changing encoder/model/target semantics inside the baseline chain;
- opponent pools, TD targets, or search; and
- requiring learned-checkpoint web play before self-play evaluation is valid.

Those research variations belong to [Milestone 7](MILESTONE7.md).
