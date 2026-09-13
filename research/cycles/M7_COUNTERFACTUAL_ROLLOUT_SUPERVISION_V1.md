# Approved step agreement: information-safe counterfactual rollout supervision

**Program:** `m7-stronger-policy-program-v1`
**Step:** 4 of 5
**Cycle:** `m7-counterfactual-rollout-supervision-v1`
**Status:** Trivial repair 2 authorized; claim completion pending
**Approved:** September 12, 2026
**Repair reauthorization:** September 13, 2026
**Prior failure report:** [`docs/M7_COUNTERFACTUAL_ROLLOUT_FAILURE.md`](../../docs/M7_COUNTERFACTUAL_ROLLOUT_FAILURE.md)
**Repair 2:** [`M7_COUNTERFACTUAL_ROLLOUT_REPAIR2.md`](M7_COUNTERFACTUAL_ROLLOUT_REPAIR2.md)
**Budget:** one CPU process; claim cutoff 7 h 45 min; hard stop 8 h
**Parent:** Step-3 mixed-data structured-v2 development recipe M

This proposal is written against the Step-3 result supplied for this decision: M v2 beat M v1 at
55.8%, beat the selected q0 parent at 82.5%, and beat random at 89.4%, with acceptable seat rows,
but failed the parent-heuristic non-regression criterion. The M arm is nevertheless the frozen
Step-4 input because Step 3 selected it by the predeclared robustness-floor rule. These percentages
are development evidence, not Step-4 outcomes.

## 1. Question, hypothesis, and isolation

**Question:** Does adding information-safe all-action counterfactual rollout supervision improve the
selected mixed-data structured-v2 recipe without changing its representation, target semantics,
optimizer, tactical envelope, or data splits?

**Treatment:** the exact Step-3 M v2 recipe plus one auxiliary loss on a fixed panel of all legal
candidate actions and fixed offline rollout targets.

**Control:** a fresh exact rerun of each Step-3 M v2 fit. The control must reproduce the three
retained Step-3 M v2 tensor digests before any treatment result is used. A digest mismatch blocks the
claim run; it is not repaired by accepting a numerically close model.

No new game collection, encoder feature, model layer, policy/belief head, rollout-time action
selection, target reweighting, class weighting, phase weighting, optimizer change, or tactical
change is allowed. The model remains one 519-input scalar-logit model. The auxiliary target is
precomputed before treatment training and is never regenerated from a treatment checkpoint.

## 2. Fixed inputs and three paired replicates

The Step-4 root seed is **`2026091204`**. All new panel, world, rollout, arena, bootstrap, and
artifact seeds derive from it with the repository's versioned `derive_seed` contract. The three MC
training shuffle/init seeds are not regenerated: they are the exact retained Step-3 M values needed
for control digest reproduction and the matched treatment horizon.

The complete machine-readable input freeze is
[`m7-counterfactual-rollout-inputs.json`](m7-counterfactual-rollout-inputs.json), fingerprint
`53ba97aed7c40a87b25f5adc1b7925e28f4aad49bd0bfb758e78b125b54181e8`. It fixes the Step-3
plan/result/validation/statistics/selection/archive identities, all three M-v2 dataset/checkpoint
files and tensor digests, epochs `26/33/31`, structured encoder/model identities, and q0–q4 plus
historical-q0 continuation checkpoint files. The executable plan must reproduce every full digest
before panel selection or control training.

Use exactly the three retained Step-3 M v2 mixed-data corpora/datasets and their complete-game or
paired-block train/validation splits. For replicate `r`:

* use the exact Step-3 M v2 q0 initialization seed and shuffle seed;
* use the exact Step-3 M v2 dataset, row order, labels, and split groups;
* initialize both control and treatment from the selected q0 tensor with the same arm-independent
  Xavier/zero-residual construction;
* create a fresh AdamW optimizer with no optimizer-state transfer;
* run one deterministic CPU thread, deterministic Torch algorithms, batch size 1024, learning rate
  `1e-3`, weight decay `1e-4`, maximum 50 epochs, patience 8, no clipping; and
* select both checkpoints by the original equal-game validation BCE on the selected-action MC rows.

The control is trained first and its checkpoint tensor digest is compared byte-for-byte with the
retained Step-3 M v2 digest for the same replicate. The frozen optimizer horizons are the retained M
`epochs_completed` values: **26, 33, and 31** for replicates 1–3. The matching treatment runs exactly
that replicate's horizon, with no treatment-specific early stopping, evaluates MC validation BCE each
epoch, and selects its best MC-validation checkpoint among those epochs. Thus the control reproduces
the Step-3 artifact exactly while treatment and control have a predeclared, matched optimizer
horizon; the only scientific factor is the auxiliary rollout gradient.

## 3. Public position panel

### 3.1 Eligibility and selection

Select **280 positions per replicate** from the replicate's Step-3 M v2 **training groups only**;
no validation-game decision, Step-3 arena decision, or prior evaluation position is eligible. A
position is selected at most once, and at most one selected position may come from one source game.
The source replay is used only to construct and verify the safe `PlayerObservation`; authoritative
hidden zones and future actions are discarded before sampling or feature construction.

Use 14 realizable fixed strata, 20 positions each:

- **play:** turn bands `1-3`, `4-6`, `7+` crossed with legal-action buckets `1-2`, `6`, `12`
  (9 strata);
- **recruit:** `(turn 1-3, unseen >=21)`, `(turn 4-6, unseen >=21)`,
  `(turn 7+, unseen <=12)`, `(turn 7+, unseen 13-20)`, and `(turn 7+, unseen >=21)`
  (5 strata).

A position is represented from the decision actor's viewpoint. Thus every recruit position has an
unknown current face-down card (`RecruitContext.known_face_down is None`); the offerer is never used
as a substitute viewpoint. Candidate rows include every semantic legal action, not only the action
recorded in the game.

Define `safe_position_identity` as SHA-256 of a canonical serialization of the complete
`PlayerObservation`, decision actor/revision, and semantic-key-sorted legal-action set. Deduplicate by
this safe identity. Within each stratum, rank eligible safe identities by
`SHA256(step4:panel:<replicate>:<stratum>:<safe_position_identity>)` and greedily select the first 20
whose source game has not already contributed a position. If the same safe observation occurs in
multiple source games, retain the lowest record fingerprint only as an offline audit locator; the
record fingerprint never affects panel rank, target worlds, policy assignments, RNG seeds, or
training order.

Claim execution is blocked if a stratum has fewer than 20 eligible safe identities after the
one-position-per-game rule. No quota relaxation or model-uncertainty selection is permitted. Record
all eligible counts, selected safe identities, audit-only source locators, and panel digest.

### 3.2 Safe latent-world sampler

Use a uniform allocation over **remaining card copies conditional only on safe public material**,
not a uniform card-type prior, learned belief, or true policy-conditioned posterior. Public actions
can signal hidden information; this deliberately exchangeable prior ignores that likelihood because
adding a belief model would be a second intervention.

Define `EXPANDED_CANONICAL_DECK` explicitly as the 38-copy list in canonical card-type order: six
copies each of Double Agent, Enforcer, Codebreaker, Daredevil, Saboteur, and Sentinel, then one
Sidekick and one Mole. Sampling operates on this expanded copy list, never the eight-type
`CARD_ORDER` alone.

For each safe panel observation:

- At play, form `U = expanded canonical deck - public recruited cards - actor's own hand`. Sample an
  ordered opponent hand without replacement for its public hand size, then reverse-Fisher-Yates
  permute the remaining copies as the hidden deck.
- At recruit, form `U = expanded canonical deck - public recruited cards - actor's own hand - known
  face-up card`. Sample the current hidden face-down copy from `U`, remove it, sample an ordered
  offerer hand without replacement for its public hand size, then reverse-Fisher-Yates permute the
  residual deck.

The sampler checks exact per-card residual counts and total cardinality before and after every
allocation. Latent zones preserve their sampled order, but the safe rollout observer canonicalizes
own-hand presentation and semantic legal-action order before every policy call. This prevents hidden
zone tuple order from changing RandomAgent or legal-action behavior while retaining exact draw order
inside the latent transition.

The sampled hidden card is a quarantined latent research variable, never a model feature or agent
input.

Do **not** weaken the engine's provenance checks or construct a forged authoritative `GameState`.
Introduce a separate trusted-offline `LatentRolloutState`/`rollout_transition_v1` type containing only
public position fields plus sampled hidden zones. Its transition code must:

1. validate card conservation, hand sizes, phase, active player, offer, revision, and legal semantic
action membership;
2. reuse the same card effects, draw counts, turn/history updates, and pure
`adjudicate_position` function as the engine;
3. omit setup seed and replay provenance entirely; and
4. never cross the live agent boundary.

The safe rollout observer projects this type to a canonicalized `PlayerObservation` without using
`GameState` or the production `observe` function. A live or frozen policy receives only that
projection, semantic-key-sorted legal actions, and its seeded RNG. The implementation must not call
`validate_state`/`_validate_provenance` on synthetic worlds and must not pass a latent state to an
encoder or policy.

## 4. Rollout target construction

### 4.1 Frozen continuation population

For every latent world, choose one fixed pair of frozen continuation policies from
this exact ten-slot population:

```text
[q0, q0, q0, q0, q1, q2, q3, q4, greedy-public-v1, random]
```

`q0` through `q4` are the retained Step-2 checkpoints; heuristic and random are the versioned
canonical policies. Every continuation policy uses the fixed terminal-offense then terminal-safety
envelope and greedy/no-epsilon deployment behavior. No Step-3 M v2 treatment or MC control is in
the continuation population. The ten-slot list gives q0 40% and every other member 10%, exactly
matching the mixed Step-2 collection marginal.

For each panel position, derive two independent seeded permutations of this ten-slot list, one for
Player One and one for Player Two. World `k=0..9` assigns the `k`th element of each permutation to
its physical seat. Each seat therefore receives q0 in exactly four worlds and every other member in
one world, while policy pairings—including self-matchups—are not restricted to one fixed offset.
The permutations depend only on replicate and safe panel identity, never candidate or latent card
contents.

This teacher matches the Step-2 **per-seat base-policy identity marginal only**. Continuation uses
greedy/no-epsilon deployment under terminal offense+safety; it does not reproduce Step-2's epsilon
behavior or joint matchup distribution.

### 4.2 Common random numbers and fixed root action

Use **10 hidden worlds × 1 continuation design = 10 rollouts per candidate**. For a panel position,
all candidates share:

- the same ten sampled latent worlds;
- the same two ten-world seat-policy permutations;
- the same initial rollout random material; and
- the same candidate-independent seed derivation.

For candidate `a`, copy the same latent world and apply `a` as the root action **without asking any
policy to select it and without applying the tactical filter**. The root action is fixed across all
worlds for that candidate; it is never reselected using the sampled hidden identity. The tactical
envelope is used only for continuation policies, leaf evaluation, and live evaluation. Therefore
provably bad but legal root actions receive valid low targets rather than being silently excluded.

Seed domains are:

```text
step4:panel:<replicate>:<stratum>:<safe-position-identity>
step4:world:<replicate>:<safe-position-identity>:<world-index>
step4:policy-permutation:<replicate>:<safe-position-identity>:<seat>
step4:rollout-rng:<replicate>:<safe-position-identity>:<world-index>:<seat>:<decision-ordinal>
step4:arena:<cell>:<replicate>
step4:nested-bootstrap:<statistic>
step4:artifact:<artifact-name>
```

No target-generating seed or training order may contain a record fingerprint, setup seed, source
action prefix, actual hidden allocation, future action, or terminal outcome. Those values remain
separate audit metadata only.

A candidate-order permutation must leave every world, policy assignment, and RNG seed unchanged.
For each continuation decision, construct a fresh deterministic policy RNG substream keyed by
`(world, physical seat, semantic decision ordinal)`. This provides common initial and per-decision
random material without reusing a mutable stream whose draw count depends on a different candidate's
path. The root policy is never called. The tactical envelope is not applied to the forced root action
but is applied to every continuation policy decision.

The immutable `counterfactual-rng-contract-v1` is part of `plan.json` and
`source-identity.json`: it uses `SEED_DERIVATION=sha256-domain-v1`,
`RNG_ALGORITHM=sha256-counter-rejection-v1`, 256-bit rejection sampling,
`EXPANDED_CANONICAL_DECK`, and reverse Fisher-Yates. Sampling without replacement removes the
selected expanded copy using `randbelow`; no language/runtime RNG is allowed. Store the contract
version and source digest in every rollout manifest.

### 4.3 Depth-nine rollout and q0 leaf value

Capture the root actor before applying the forced root action. Continue each latent world for at most
**9 semantic actions inclusive of the root**: the root plus at most eight continuation decisions.
After every action, return an exact terminal value when the pure adjudicator produces an outcome.
Otherwise stop at the depth limit and evaluate the leaf.

Use the frozen selected **q0 parent** under terminal offense+safety as the leaf evaluator. It predates
the Step-3 M training groups and is cheaper than structured v2. This makes the target a finite-depth,
q0-bootstrapped teacher rather than a game-theoretic value.

At a depth leaf:

1. safely observe the current actor from `LatentRolloutState`;
2. apply the fixed terminal-offense/safety envelope;
3. score the complete allowed legal set with q0;
4. take the deterministic greedy action value `sigmoid(logit)` without consuming RNG; and
5. convert to root-actor viewpoint: use the value if leaf actor equals root actor, otherwise
   `1 - value`.

A terminal result is exact `0/1` from the root actor's viewpoint. A latent position that completes
turn 19 without a terminal result is an implementation error, not a learned leaf; no turn-20
observation may reach q0 or v2. Record terminal counts, depth-leaf counts, depth distribution, and
mechanical-cap errors. The total depth-leaf fraction across all target samples must be at most 50%; a
larger fraction is a frozen scientific-integrity failure and does not authorize increasing depth.

The rollout target is the fixed ten-world average

```text
R_hat(o,a) = mean over k=0..9 of
             [1 if terminal winner is root actor,
              0 if terminal winner is not root actor,
              frozen-q0 root-actor win probability at the depth leaf].
```

`R_hat` is a deterministic finite-design estimate under the declared exchangeable allocation prior,
balanced policy permutations, depth-nine transition rule, and q0 leaf. It is not a true posterior
expectation, iid population estimate, game-theoretic value, belief probability, ranking label, or
selected-action label.

Retain complete target samples, seeds, terminal/leaf flags, and digests for every rollout. Retain
full hidden-zone transcripts only for one canonical safe position and semantic-first candidate per
stratum across ten worlds: **3 replicates × 14 strata × 10 = 420 audit trajectories**. Full hidden
transcripts for every candidate are prohibited as unnecessary I/O and leakage surface.

## 5. Auxiliary objective and training schedule

For each panel position, retain one rollout row for every legal semantic action and the target
`R_hat(o,a)`. Encode rows with the unchanged 519-feature safe structured-v2 encoder. The treatment has
one scalar output and uses

```text
L_MC   = mean selected-action BCE-with-logits on the ordinary M training batch
L_roll = mean over positions [mean over all legal candidates BCE-with-logits(z(o,a), R_hat(o,a))]
L_step = L_MC + 0.20 * L_roll
```

The position mean precedes the candidate mean so positions, rather than high-action-count play
states, determine auxiliary weight. Soft targets in `[0,1]` are passed directly to
`BCEWithLogitsLoss`; no hardening, clipping, ranking margin, or target calibration is applied.

Use the exact MC optimizer-step count and deterministic MC minibatch order from Step 3. Define one
canonical panel order: increasing stratum index, then panel-selection hash, then safe position
identity. Set the rollout cursor to zero before epoch 1; use complete 128-position shards, advance the
cursor by 128 at each MC step, wrap at the end, and never reshuffle between epochs. A shard therefore has a variable number of candidate rows but computes
`mean_position(mean_candidate_BCE)` exactly. This keeps the optimizer, number of updates, shuffle
seed, and MC examples matched to the control horizon `E_r`. The treatment differs only by the
declared `+0.20 L_roll` gradient.

No treatment loss sees the recorded chosen action specially inside `L_roll`; all legal candidates
are present. No auxiliary target is materialized for validation games.

## 6. Claim-generating evaluation

Use new setup-seed domains disjoint from all Step-2/Step-3 corpora, arenas, panel source games, and
prior retained runs. Each arena has matched two-game seat-swapped blocks, preserving logical-agent
RNG identities across the swap. Apply the fixed offense+safety envelope to q0, controls, treatments,
q1-q4, and the fixed M-v2 leaf/reference policies. Canonical heuristic and random retain their
versioned unwrapped evaluation policies. In the table below, **each candidate** explicitly means
both `T_r` (rollout treatment) and `C_r` (matched MC control) for the same replicate `r`.

Per replicate, use this exact schedule:

| Comparison | Paired blocks per replicate |
| --- | ---: |
| `T_r` vs `C_r` direct | 500 |
| `T_r` vs selected q0 parent; `C_r` vs selected q0 parent | 500 each |
| `T_r` vs heuristic; `C_r` vs heuristic | 300 each |
| selected q0 parent vs heuristic reference | 300 |
| `T_r` vs random; `C_r` vs random | 200 each |
| `T_r` vs historical q0; `C_r` vs historical q0 | 200 each |
| `T_r` and `C_r` vs each enveloped q1, q2, q3, q4 | 100 each candidate/opponent |
| **Total physical games** | **8,000 per replicate / 24,000 total** |

The treatment-control direct arena is the primary comparison. The parent/heuristic cells share the
same setup block within a replicate so the treatment-minus-parent heuristic difference is a matched
contrast. q1-q4, historical-q0, and equal-opponent results are descriptive robustness diagnostics,
not substitute promotion statistics.

## 7. Statistics and frozen gates

Use the existing deterministic nested-bootstrap contract: 20,000 resamples, outer resampling of
the three training replicates followed by inner resampling of matched paired setup blocks, nearest
rank indices 499 and 19,499, and materialized outer draws before any inner RNG. Use separate named
inner streams per opponent cell. For the heuristic non-regression contrast, do **not** use independent
streams: within each outer replicate draw, resample one identical aligned block-index vector for both
`T_r-vs-heuristic` and `q0-vs-heuristic`, then compute
`Delta_H = WR(T_r,H) - WR(q0,H)`. Use the same aligned-block rule for `C_r-vs-heuristic` as a
secondary diagnostic. Report replicate point estimates, both seats, score margin, turns/decisions,
terminal reasons, throughput, and rollout target diagnostics.

Predeclared estimands:

1. treatment-minus-MC-control direct paired win rate;
2. treatment and control versus q0 parent;
3. treatment-minus-parent and control-minus-parent heuristic contrast on the shared reference
   blocks;
4. treatment versus random, historical q0, and q1-q4; and
5. equal-weight descriptive macro over q0 parent, heuristic, random, historical q0, q1, q2, q3, q4.

The rollout treatment advances as a **Step-5 eligible recipe family** only if all conditions hold:

* nested lower endpoint for treatment versus matched MC control is strictly above 50%;
* at least two of three direct replicate treatment point estimates exceed 50%;
* nested lower endpoint for treatment versus q0 parent is above 50%;
* nested lower endpoint for treatment versus random is above 50%;
* nested lower endpoint of treatment-minus-parent heuristic is strictly above -5 percentage points;
* the minimum over every indexed treatment seat row
  `(replicate, cell, physical_seat)` with `replicate ∈ {1,2,3}`, `cell ∈ {T-vs-C, T-vs-q0,
  T-vs-heuristic, T-vs-random}`, and `physical_seat ∈ {PLAYER_ONE, PLAYER_TWO}` is at least 45%;
* every treatment record has zero executed avoidable provable losses and zero missed guaranteed
  current-turn wins under the fixed envelope; and
* all source, split, panel, sampler, rollout, replay, digest, schedule, and validator checks pass.

A failed or interval-crossing gate is `does_not_advance`/`inconclusive_does_not_advance` according to
the direct interval. No extra panel positions, rollouts, arena blocks, or training replicates may be
added after inspecting results.

### Step-5 disposition

If the recipe passes every gate, all three treatment checkpoints enter the independent Step-5 league
as one rollout-supervised recipe family; no replicate is cherry-picked. The three MC-only controls
remain paired references and are retained, but do not become a promoted champion merely because the
treatment passed.

If any gate fails, no treatment checkpoint is entered as a Step-5 advancement candidate. Retain all
control/treatment artifacts and report the failure. Step 5 may still run its already authorized
league over q0, q1-q4, the tactical q0 variant, and the predeclared surviving Step-3 M-v2 recipe as
descriptive entries; it may not reinterpret a failed rollout recipe as a promotion candidate.
Step 4 never changes the selected champion or web default.

## 8. Artifacts and deterministic validation

Retain outside Git, with checksums:

```text
plan.json
source-identity.json
inputs/step3-M-reference.json
control-reproduction/{replicate}/summary.json
panel/{replicate}/positions.jsonl.gz
panel/{replicate}/manifest.json
rollouts/{replicate}/targets.npz
rollouts/{replicate}/manifest.json
rollouts/{replicate}/trusted-world-transcripts.jsonl.gz
training/{control,treatment}/{replicate}/history.json
checkpoints/{control,treatment}/{replicate}/
arenas/<cell>/records/ and report.json
statistics.json
safety-report.json
selection.json
result.json
checksums.json
```

Panel and model-input artifacts contain only safe observations/features, semantic actions, public
position metadata, and audit provenance. Trusted rollout transcripts may contain sampled latent
cards and outcomes, but are never passed to training, live agents, the browser, or validators that
are intended to model the safe boundary.

The immutable role map in `inputs/step3-M-reference.json` must name, for every replicate: `q0-parent`
(as leaf and external reference), `q1`–`q4`, `heuristic`, `random`, `historical-q0`, retained
`M-v2`, reproduced `M-v2-matched-control`, and `M-v2-rollout-treatment`. The selected q0 parent is
the only leaf; neither retained M-v2 nor the treatment may act as leaf. Every tactical gate includes
all treatment arena records and excludes only non-candidate parent/reference records.

An independent validator must, without importing the production rollout sampler, rollout loss, or
production statistics/selection entry points:

* verify exact Step-3 M input/checkpoint identities and reproduce every control tensor digest;
* verify train-only panel membership, all 14 quotas, position/action counts, and panel fingerprints;
* reconstruct safe observations and independently sample the same copy-weighted worlds;
* verify latent-state card conservation and transition equivalence against real engine states on
  fixtures, without invoking engine provenance for synthetic worlds;
* recompute all target rows and common-random-number assignments bit-for-bit;
* verify root actions are fixed and continuation/leaf policies receive only safe observations;
* verify unchanged v2 feature digests, treatment/control training settings, MC validation selection,
  checkpoint tensor digests, arena schedules, tactical audits, and nested bootstrap endpoints; and
* verify Step-5 disposition from `selection.json` without reselecting a replicate.

Required tests include safe-position identity and target-seed invariance under source setup/hidden/future
action changes, expanded-copy marginal and joint frequency checks, exact policy marginals and varied
pairings, canonical hand/legal-action order, recruiter hidden-card non-leakage, root-action fixation,
per-decision common-random-number invariance, latent/authoritative transition equivalence, active-tie
and deck-exhaustion fixtures, q0 leaf viewpoint inversion, zero mechanical turn-cap errors,
depth-leaf threshold accounting, position-balanced rollout loss, and treatment with
`lambda_roll=0` reproducing control tensors after every optimizer step/epoch.

The independent validator must also have static source guards and mutation tests for card
conservation, hidden-card leakage, policy assignment, root reselection, draw order, terminal tie,
leaf inversion, candidate weighting, target values, arena reports, and Step-5 disposition.

## 9. Runtime envelope

The fixed panel contains **280 positions per replicate**, at most 1,400 legal candidate rows per
replicate, and 10 depth-nine rollouts per candidate: at most **42,000 candidate rollouts** and
378,000 applied semantic actions/leaf evaluations production-side, with one independent
recomputation. A timed smoke must measure the actual VM before claim execution, including latent
transition/leaf throughput, independent target recomputation, audit-transcript compression, and the
24,000-game arena total.

The declared allocation is:

| Work | Maximum allocation |
| --- | ---: |
| input audit, control reproduction, panel extraction, timing preflight | 20 min |
| production rollout targets, at most 420k transition/leaf work units | 105 min |
| three treatment fits and checkpoint audits | 15 min |
| 24,000-game arena, record I/O, tactical replay | 80 min |
| production statistics, artifacts, checksums | 30 min |
| independent input/panel/control/latent checks | 15 min |
| independent target recomputation, at most 420k work units | 105 min |
| independent arena replay/aggregation | 30 min |
| independent statistics, selection, checksums/result | 30 min |
| reserve | 35 min |
| **claim cutoff** | **465 min (7 h 45 min)** |

Before freezing claim source, a production-plus-independent smoke must sustain at least **67 latent
transition/leaf work units per second**, including serialization, and project the complete run below
465 minutes. Depth-leaf fraction must project at or below 50%, mechanical turn-cap errors must be
zero, and all 14 panel strata must meet quota. Failure is `implementation_blocked`; it does not
permit reducing panel positions, worlds, depth, games, or replicates.

## 10. Risks and fixed mitigations

* **Synthetic-state provenance:** arbitrary sampled hidden allocations cannot honestly be represented
  as a replay from the original setup seed. The separate latent rollout state deliberately avoids
  forged `GameState` objects and has independent structural/transition validation.
* **Hidden-information leakage:** the sampler is observation-only, discards the true source hidden
  allocation, and exposes only safe projections to policies/encoders. Trusted omniscient transcripts
  are quarantined.
* **Strategy fusion:** the root action is forced identically across worlds; continuation policies
  choose only from each seat's safe observation; no branch uses the root player's omniscient world.
* **Teacher-policy bias:** rollout values are conditional on a fixed, weakly heterogeneous frozen
  population and a q0 leaf. A positive result is only evidence for this declared teacher, not
  general game value.
* **Uniform-prior misspecification:** the copy-weighted unseen-card prior is not a learned belief and
  may be wrong about strategy-induced correlations. This is intentional: no belief head existed in
  Step 3, and introducing one would violate isolation. Report prior sensitivity descriptively only;
  do not tune it after results.
* **Leaf bias:** depth-nine paths use selected q0, which is independent of the Step-3 M panel-source
  training but known to be strategically limited. Report terminal/leaf fractions and fail if leaf
  use exceeds 50%; no depth or leaf substitution is allowed after results.
* **Panel overfitting:** the panel is train-only and fixed before treatment fitting; it is not a
  validation or arena metric. Reuse of the same panel across epochs is declared auxiliary training,
  not evidence of held-out strength.
* **Auxiliary domination:** `L_MC` remains coefficient one, `L_roll` is position-balanced and fixed
  at weight 0.20, and MC validation checkpoint selection uses the predeclared matched horizon `E_r`;
  no lambda or stopping-horizon tuning is permitted.
* **Small replicate count and non-transitivity:** three complete paired replicates are the unit of
  recipe variation; external macros are descriptive and cannot replace direct or robustness gates.
* **Runtime:** control digest reproduction precedes treatment claims; timed smoke and a hard cutoff
  prevent an over-budget partial run from becoming evidence.

**Interpretation limit:** even a passing result supports only the mixed-data structured-v2 recipe plus
this fixed rollout teacher under the declared CPU, policy population, exchangeable public-count
prior, depth-nine/q0-leaf rule, and development league. It does not establish a correct learned
belief, game-theoretic values, optimal play, or a universal benefit from rollout supervision.

## 11. Frozen implementation and preflight

- `b20699b`: safe panel identity/selection, latent world state/sampler/observer/transitions,
  candidate-independent continuation assignments/RNG, depth-nine q0-leaf targets, target artifacts,
  and paired rollout-loss training primitives.
- `1b381b9`, `bcc0c04`, `6337f8f`, `6c46d3b`: resumable experiment, sharded targets, control
  reproduction, claim/smoke CLI, and validator-local numeric latent target recomputation.
- `a739db3`, `c9a69d1`, `e2029e9`: matched claim arena groups, complete local claim statistics,
  corrected production/validator projections, and serialized report compatibility.
- `cb0e045`: exact result/cardinality/reference/checksum validation and independent treatment
  tactical replay.

A retained-input one-position-per-stratum smoke at `/tmp/agent-avenue-step4-smoke-3ac1e6f` passed
production and independent validation. Production target throughput was 159.93 units/s; independent
recomputation was 160.32 units/s. Maximum production leaf fraction was 46.86%, validator pooled leaf
fraction 45.62%, and mechanical cap errors were zero. The corrected full combined projection is
204.73 minutes, below the 465-minute claim cutoff.

The immutable preflight input is retained at
`artifacts/preflight/m7-counterfactual-rollout-preflight-cb0e045.json`, artifact fingerprint
`557d16a9ee239500c7f9064311d16684b65b10b226b2a2430f6f723660088c7d`, bound to package code
fingerprint `81913c1e52a22776fde97661936b7793f51f6c6882ce418aa76220a03de258d7` and input manifest
fingerprint `53ba97aed7c40a87b25f5adc1b7925e28f4aad49bd0bfb758e78b125b54181e8`.

Ruff, strict mypy, and 315 tests pass. A final fresh preflight review is GO.
