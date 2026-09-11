# Approved step agreement: population replay v1

**Program:** `m7-stronger-policy-program-v1`
**Step:** 2 of 5
**Cycle ID:** `m7-population-replay-v1`
**Status:** Approved; implementation pending
**Approved:** September 11, 2026
**Prerequisite:** [`M7_TERMINAL_OFFENSE_CONFIRM_V1.md`](M7_TERMINAL_OFFENSE_CONFIRM_V1.md) completed with structural adoption

## Question

With encoder, model, selected-action Monte Carlo target, optimizer, initialization, exploration rate,
and exact tactical envelope fixed, does collecting replay from a mixed policy population produce a
stronger and more robust model-v1 candidate than collecting the same number of q0-only self-play
games?

This step isolates **collection distribution**. It does not add public history, candidate-consequence
features, ranking, TD, counterfactual targets, search, or a larger model.

## Fixed behavior envelope

Every corpus behavior policy is composed as:

```text
TerminalOffense(
    TerminalSafety(
        EpsilonGreedy(base_policy, epsilon=1/5)
    )
)
```

The adopted offense and safety filters run before exploration, so random exploration cannot restore
a vetoed loss or skip a guaranteed current-turn win. Raw q0–q4 checkpoints are wrapped once; no
historical shield configuration is nested twice.

## Corpus arms

Run three paired corpus/training replicates. Each arm contains **2,000 paired setup blocks / 4,000
seat-swapped games** per replicate.

### Control: q0-only

Both logical policy slots use the selected q0 checkpoint under the fixed envelope and epsilon.
Logical lane A and B retain independent RNG streams across the seat swap.

### Treatment: mixed population

Deterministically balance 4,000 logical policy slots per replicate:

| Base policy | Logical slots | Weight |
| --- | ---: | ---: |
| q0 | 1,600 | 40% |
| q1 | 400 | 10% |
| q2 | 400 | 10% |
| q3 | 400 | 10% |
| q4 | 400 | 10% |
| `greedy-public-v1` | 400 | 10% |
| random | 400 | 10% |

A declared assignment RNG permutes these slots and pairs them into 2,000 logical matchups. Each
logical policy appears once in each physical seat through the paired seat swap, so every per-seat
marginal exactly matches the table. Self-matchups are allowed and retained.

Historical q0 and safety-only/offense-only diagnostics are evaluation-only. All live policies receive
only safe public observations and semantic legal actions.

## Pairing, seeds, and holdout

- Root seed: `2026091102`.
- Replicate IDs: `replicate-1`, `replicate-2`, and `replicate-3`.
- Derive corpus/setup, population assignment, agent lane, split, initialization/shuffle, arena, and
  bootstrap seeds by named `sha256-domain-v1` domains.
- Within a replicate and pair index, control and treatment share the same setup seed and physical
  seat schedule. Their logical agent lane RNG identities are arm-independent; trajectories may
  diverge only because the assigned base policies differ.
- Replicate setup families are mutually disjoint.
- Before collection, recursively scan every completed corpus under `runs/`, excluding the current
  output subtree, and require zero normalized game-config/setup-seed overlap for all training and
  development blocks.
- Total fresh training games: **24,000**.

The generated immutable plan must materialize every derived seed, assignment, matchup count, policy
configuration, and checkpoint fingerprint before the first game.

## Dataset and training

For each arm and replicate:

- deterministic 90/10 split by complete paired setup block;
- matched control/treatment split assignment using the same split seed;
- require every treatment population member in both train and validation partitions;
- retain policy identity as provenance only, never as a model feature;
- initialize from the same selected q0 tensor weights;
- create a fresh optimizer with no optimizer-state transfer; and
- use the replicate's paired training shuffle seed for control and treatment.

Keep exactly:

- encoder `candidate-public-v1`;
- model `candidate-mlp-v1` (87→128→1, 11,393 parameters);
- acting-player eventual terminal outcome for the selected action;
- binary cross-entropy with logits;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- batch size 1,024;
- maximum 50 epochs, patience 8, equal-game validation selection;
- deterministic CPU algorithms and one Torch thread; and
- no sample/loss weighting, ranking labels, TD targets, auxiliary heads, or model-capacity changes.

Equalize games, not decisions. Report trajectory length, decision/sample count, policy/phase/card
action coverage, target balance, calibration, and throughput as consequences of the collection arm.

## Fresh development evaluation

Use fresh, replicate-specific setup domains disjoint from every corpus and prior retained run. Apply
the fixed offense+safety envelope to q0, q1–q4, and both trained candidates. Keep heuristic, random,
and historical q0 in their canonical historical forms.

Per replicate:

| Arena | Paired blocks |
| --- | ---: |
| Treatment candidate versus matched control candidate | 500 |
| Each candidate versus fixed-envelope q0 parent | 500 each |
| Each candidate versus canonical heuristic | 300 each |
| Fixed-envelope q0 parent versus heuristic on the same block | 300 |
| Each candidate versus canonical random | 200 each |
| Each candidate versus historical q0 | 200 each |
| Each candidate versus fixed-envelope q1, q2, q3, q4 | 100 each |

This is 7,400 games per replicate and **22,200 development games** total. q1–q4 stress cells are
descriptive because those policies are treatment-pool members and correlated descendants. No
locked-final block opens in step 2.

Control/treatment arenas against a shared opponent use the same setup blocks, seats, opponent RNG
identity, and candidate lane RNG identity. Arena records, safety/offense diagnostics, terminal
reasons, score margins, game lengths, seats, and throughput are retained.

## Statistics

Use a deterministic 20,000-resample nested bootstrap:

1. outer resampling over the three paired corpus/training replicates; and
2. inner resampling over matched paired setup blocks within each selected replicate.

Do not pool individual games, decisions, or checkpoints as independent observations.

Primary statistic: treatment-candidate win rate against its matched control candidate. Also report:

- each replicate separately;
- treatment and control versus q0 parent;
- treatment-minus-control differences against heuristic, random, historical q0, and q1–q4;
- treatment-minus-parent heuristic difference on the shared block;
- absolute and equal-opponent descriptive macros; and
- between-replicate variation.

## Step-2 decision

Call the mixed-population treatment an **advancing model-v1 collection recipe** only if all hold:

1. nested 95% lower endpoint for treatment versus control is strictly above 50%;
2. at least two of three direct replicate point estimates favor treatment;
3. nested treatment-versus-q0-parent lower endpoint is above 50%;
4. nested treatment-versus-random lower endpoint is above 50%;
5. nested treatment-minus-parent heuristic lower endpoint is above -5 percentage points;
6. nested treatment-minus-control heuristic lower endpoint is above -5 percentage points;
7. no candidate seat point estimate is below 45%;
8. zero executed avoidable immediate losses and zero missed guaranteed wins for every enveloped
   candidate; and
9. replay, schedule, seed, source, split, checkpoint, and compatibility checks pass.

Failure to satisfy every condition is `does_not_advance`; an interval crossing 50% is
`inconclusive_does_not_advance`. Do not add games, change weights, or select a favorable replicate.

Step 3 proceeds regardless of this decision. Both fixed control and population corpora/datasets and
all six model-v1 checkpoints are retained. Step 3 will predeclare a structured-model comparison that
uses both data arms, avoiding post-result corpus selection and allowing data×architecture interaction
analysis. A scientific-integrity failure after the one allowed repair blocks step 3.

No step-2 result changes the selected champion or web default.

## Validation and limits

- Add one resumable population experiment runner and one independent same-source validator.
- Validate exact assignment marginals, paired setup/split alignment, holdout exclusion, nested
  bootstrap, control/treatment initialization identity, training configuration, checkpoint lineage,
  replay, tactical invariants, and every arena aggregate.
- Obtain a fresh-context result review and checksum archive before completion.
- Claim cutoff: **7 hours 45 minutes**; whole-step CPU budget: **8 hours**.
- One substantial process at a time; at most two implementation workers in isolated worktrees.
- At most one operational repair/retry, which may not change policies, weights, games, seeds,
  training, arenas, statistics, or decision criteria.
- Generated evidence belongs under `runs/m7-population-replay-v1/`; compact source, agreement,
  progress, and result notes are committed.

## Out of scope

- reusing September evaluation records as training data;
- public-history or structured-model changes;
- heuristic ranking pretraining;
- TD, policy gradients, search, belief-state rollouts, or counterfactual targets;
- promotion or web deployment; and
- interpreting q1–q4 stress cells as independent expert validation.
