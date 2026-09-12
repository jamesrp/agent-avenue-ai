# Approved step agreement: structured public-history model v2

**Program:** `m7-stronger-policy-program-v1`
**Step:** 3 of 5
**Cycle:** `m7-structured-model-v2`
**Status:** Approved and implementation-frozen; claim run pending
**Approved:** September 12, 2026
**Budget:** one CPU process, claim cutoff 7 hours 45 minutes, hard wall-clock limit 8 hours

## 1. Question and fixed interpretation

Step 3 asks whether an information-safe structured representation improves the selected-action
terminal Monte Carlo value model, and whether the answer depends on the Step-2 collection arm.
It is a predeclared **2 × 2 data × architecture study**:

| Data arm | Architecture v1 | Architecture v2 |
| --- | --- | --- |
| q0-only control | retained Step-2 checkpoint C1–C3 | new V2-C1–C3 training |
| mixed population | retained Step-2 checkpoint M1–M3 | new V2-M1–M3 training |

There are three paired training replicates in each cell. The six v1 checkpoints and six
corpora/datasets are fixed Step-2 artifacts; the six v2 fits cover both data arms, each using its
corresponding retained dataset. The mixed arm is not selected after observing the Step-2 result.

The Step-2 result entering this design is retained as evidence, not as a changed gate: population
v1 won 62.1% of matched direct games, beat q1–q4 in the declared stress cells, and improved the
heuristic comparison relative to control, but did not advance because the treatment-minus-parent
heuristic interval had lower endpoint -5.056 percentage points. The frozen all-candidate seat
criterion reached 33.8%; this was the **control** candidate's derived Player-One rate in replicate-3
direct treatment-versus-control play, not a treatment seat collapse. Treatment's lowest observed
seat rate was 54.0%. No Step-3 corpus, checkpoint, or evaluation cell is removed because of that
result.

### Fixed quantities

* behavior corpora: exactly the six retained Step-2 corpora;
* labels: selected-action, acting-player eventual terminal win/loss;
* loss: terminal Monte Carlo binary cross-entropy with logits;
* model deployment: greedy candidate logit with the already adopted terminal-offense plus
  terminal-safety envelope;
* no counterfactual rollout labels, ranking labels, TD/lambda targets, policy labels, belief target,
  auxiliary head, search, or new behavior collection;
* no locked-final arena is opened in Step 3; and
* no champion, web default, or q0 parent is changed by this step.

The v2 feature constructor accepts only `PlayerObservation` and one member of its semantic legal
action tuple. It must never accept `GameState`, a successor state, the true face-down card, the
opposing hand, or deck order.

## 2. Frozen Step-2 inputs and compatibility

Step 3 accepts only the validated Step-2 archive and these identities:

- Step-2 claim source: `fde19b5d3c327e539c29913a973b5e4d76ffff5b`;
- Step-2 plan: `a85c4d352386be2512a71dd08326abe45a3f80cca5a762556c749e3f02408a31`;
- Step-2 result: `aaf30bf3b6939f964a8c69a29a60e911859c253ca5e20517091f4b7460c954f2`;
- Step-2 repaired validation: `1930ffc5df1378bbc0daad484d1140707553b3f4b69fe7d4c869abf2589905e2`;
- Step-2 split alignment: `9653dfdf1e7cb7f401af753ccfe8e90ce270a7fc12d97c98f097860c4c774552`;
- selected q0 checkpoint: `b511ae162b794da6450fd3151762d98b4e5c8475c1af23e5b9289d0140a0a06d`;
- selected q0 tensor: `b1562237220e3a59b1e514327f3a5acc3d60767676befc69970f88ef98253767`;
- rules fingerprint: `877731565922c8e65245f78a8bb71995cfd26edd81dab13fac2520f970695d09`.

| Replicate | Arm | Corpus | Dataset | NPZ SHA-256 | Manifest SHA-256 | Train-pair set | Validation-pair set | v1 checkpoint | v1 tensor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | control | `39f3cd20…4652` | `7e04f406…1018` | `7ccbaf51…7f51` | `287cac2b…de8` | `4342af38…42ab` | `22c445ea…1597` | `1156e315…954e` | `1418cb63…15c2` |
| 1 | mixed | `d34735fc…b970` | `2501a7f9…817b` | `8fbc2d84…031e` | `811e5910…5402` | `4342af38…42ab` | `22c445ea…1597` | `47721bb0…55df` | `25261fdf…f962` |
| 2 | control | `98810c91…4af0` | `1e9cd120…17ad` | `d67d4f56…b4a0` | `80b1b689…712b` | `31782d05…06cd` | `079621db…4c2c` | `48bd9bef…fd61` | `14f1cf8f…871c` |
| 2 | mixed | `8dbeec84…102e` | `95ab4df9…28ba` | `d9503d58…a604` | `33ccd7fa…4a67` | `31782d05…06cd` | `079621db…4c2c` | `00014dd8…93e7` | `1e824ea5…f48c` |
| 3 | control | `3388f433…a2ad` | `7dfaa39d…d608` | `a7d3af5e…6c40` | `4f5a4c9a…b8ca` | `ce1509b2…d65c` | `fb01b2c6…cdcf6` | `da959934…28c8` | `d837cdb0…c7f6` |
| 3 | mixed | `1a928a07…0a78` | `6eef42c1…d8a9` | `6963fd0f…af2b` | `dcb16b54…2654` | `ce1509b2…d65c` | `fb01b2c6…cdcf6` | `cc882287…7f52` | `f10b74bd…78b1` |

The complete machine-readable input freeze is
[`m7-structured-model-v2-inputs.json`](m7-structured-model-v2-inputs.json), fingerprint
`534ea9aaad0fe9033d568ae6ef945828649ad4030899e004175a3057ad4aa186`. The generated Step-3 plan
must reproduce every full digest in that file rather than relying on the abbreviations above.

Adding v2 changes the current package code fingerprint. Compatibility therefore requires explicit
version dispatch rather than silently weakening checks:

- retain immutable v1 dataset/checkpoint loaders and a v1 inference adapter for the six controls;
- define separate v2 dataset, encoder, model, and checkpoint schema versions;
- load Step-2 records only after verifying their frozen corpus/file identities, then replay them with
  current rules and `verify_code=False` because the code fingerprint is historical;
- require the current rules fingerprint and semantic replay result to match every record;
- require the engine, observation, and fixed tactical-envelope semantics to differ from Step-2 only
  through an explicit audited allowlist; and
- reject any guessed migration or incompatible artifact loudly.

## 3. Safe encoder v2

### Version, width, and ordering

The encoder is `candidate-public-structured-v2`, width **519**, with a fingerprint over version,
feature names/order, card order, scales, history window, consequence definitions, and width.
The vector is the concatenation:

```text
[ candidate-public-v1 (87),
  public-history-v2 (314),
  play-consequence-v2 (64),
  recruit-consequence-v2 (54) ]
```

The first 87 values are byte-for-byte the v1 safe encoder output. The inactive consequence block is
all zeros: play candidates populate only the play block, and recruit candidates populate only the
recruit block. This makes phase routing auditable and prevents a play branch from consuming recruit
features or vice versa.

All scalar values are finite `float32`; one-hot values are exactly 0/1. Card one-hots use the
existing immutable `CARD_ORDER`. No feature uses an action index, tuple position, setup seed, RNG
state, code revision, hidden card identity, or authoritative successor.

### 3.1 Base state/action block: v1 unchanged (87)

Use `candidate-public-v1` exactly, including its viewpoint-relative public state, own hand counts,
public recruited thresholds, safe unseen counts, current face-up card, candidate play cards, and
candidate recruit slot. Do not add a second copy of any v1 feature under a different name.

### 3.2 Public history block (314)

Use the last **8 completed public turns**, oldest to newest, left-padded with zero events. The current
incomplete play/recruit decision is never placed in history. If more than eight turns have completed,
keep the most recent eight and expose the truncation through the two summary features below.

Each event is 39 values, in this exact order:

| Event group | Width | Definition |
| --- | ---: | --- |
| `valid` | 1 | 1 for a real completed turn, 0 for left padding |
| `active_player_relative` | 2 | one-hot self/opponent for `CompletedTurn.active_player` |
| `face_up` | 8 | public completed-turn face-up card |
| `face_down` | 8 | public completed-turn face-down card; it is public only because the turn is complete |
| `chosen_slot` | 2 | public face-up/face-down recruit choice |
| `opponent_recruited` | 8 | card assigned to the historical recruiter |
| `active_recruited` | 8 | card assigned to the historical offerer |
| `self_score_change` | 1 | viewer-relative score change, clipped to `[-3, 6]` and divided by 6 |
| `opponent_score_change` | 1 | opponent-relative score change, clipped to `[-3, 6]` and divided by 6 |

The two summary values are:

* `history_length_fraction = len(history) / 19`; and
* `history_truncated_fraction = max(len(history) - 8, 0) / 19`.

The event card fields retain the semantic `active`/`opponent` roles in the public record; the actor
relative one-hot tells the model whether the historical offerer was the current viewer. Score
changes are reoriented to current viewer/opponent before encoding. This block contains no current
hidden offer card and no private historical hand information.

### 3.3 Play-consequence block (64)

A play candidate is an offer of two cards known to the current offerer. It is safe to describe the
publicly possible consequences of the next recruit choice without executing either branch. For each
of the two hypothetical **next recruit slots**, in block order `recruit-face-up` then
`recruit-face-down`, emit 30 values:

| Branch group | Width | Definition |
| --- | ---: | --- |
| `self_card` | 8 | card the current offerer would receive in this branch |
| `opponent_card` | 8 | card the next recruiter would receive |
| `self_effect` | 3 | one-hot `score`, `win`, `lose` from current public recruited counts |
| `opponent_effect` | 3 | same for the next recruiter |
| `self_terminal` | 2 | exact branch outcome one-hot immediate win / immediate loss, otherwise 0/0 |
| `opponent_terminal` | 2 | same exact outcome from the opponent viewpoint |
| `self_score_delta` | 1 | immediate score effect divided by 6; 0 for win/lose |
| `opponent_score_delta` | 1 | same |
| `self_resulting_count` | 1 | resulting count divided by that card's canonical copies |
| `opponent_resulting_count` | 1 | same |

The branch assignment follows the engine's public semantics: if the recruiter takes face-up, the
recruiter gets candidate `face_up` and the offerer gets candidate `face_down`; if the recruiter takes
face-down, the assignments reverse. Effects are looked up from public recruited counts and card
definitions only. For the terminal bits, construct the resulting public scores/tableaux, public deck-
exhaustion fact, active player, and next-player hand size, then call only the pure engine
`adjudicate_position` evaluator. Do not construct or transition an authoritative `GameState`, inspect
draw identities, or perform a rollout.

The four remaining values are candidate-level consequences:

1. `post_play_hand_without_refill_fraction = (own_hand_size - 2) / 4`;
2. `post_play_refill_count_fraction = min(4 - (own_hand_size - 2), remaining_deck_count) / 4`;
3. `post_play_hand_size_fraction = (own_hand_size - 2 + refill_count) / 4`; and
4. `offer_same_card` (1 when the legal offer contains two copies of the same card, else 0).

Card identities of the two offered cards are already present in the v1 candidate block; these
features expose their public branch consequences rather than replacing the action representation.
Draw identities are never represented.

### 3.4 Recruit-consequence block (54)

At a recruit decision the current face-up card is public and the face-down card is unresolved. For
the candidate slot, emit the following exact groups:

| Group | Width | Definition |
| --- | ---: | --- |
| `known_card_to_self` | 8 | one-hot only if the candidate gives the known face-up card to self |
| `known_card_to_opponent` | 8 | one-hot only if it gives the known face-up card to opponent |
| `hidden_to_self`, `hidden_to_opponent` | 2 | one bit for the unresolved recipient |
| `known_effect_self`, `known_effect_opponent` | 6 | 3-way score/win/lose one-hot for the known-card recipient; zero for the unresolved recipient |
| `guaranteed_terminal_self`, `guaranteed_terminal_opponent` | 4 | exact immediate win/loss bits only when the same outcome holds across every public-consistent hidden identity |
| `known_score_delta_self`, `known_score_delta_opponent` | 2 | known immediate score effect divided by 6 |
| `known_resulting_count_self`, `known_resulting_count_opponent` | 2 | known card resulting count divided by canonical copies |
| `known_card_recipient_prior_count` | 1 | known card recipient's public prior count divided by canonical copies |
| `hidden_effect_support_self`, `hidden_effect_support_opponent` | 6 | for each score/win/lose category, fraction of safe unseen card copies that could cause that category |
| `hidden_score_minmax_self`, `hidden_score_minmax_opponent` | 4 | min/max score effect over card types with nonzero safe unseen count, each divided by 6; zero if recipient is known |
| `hidden_card_mass_self`, `hidden_card_mass_opponent` | 2 | `1 / max(total safe unseen count, 1)` for unresolved recipient, else 0 |
| `hidden_effect_support_bits_self`, `hidden_effect_support_bits_opponent` | 6 | at-least-one-support bits for score/win/lose |
| `safe_unseen_pool_fraction` | 1 | total information-visible unseen copies divided by 38 |
| `candidate_terminal_support` | 2 | at-least-one public-consistent hidden identity yields an immediate self win / self loss |

The widths sum to 54. Current face-up identity and candidate slot are not duplicated because they
already appear in the unchanged v1 prefix. The hidden support fields are **possibility envelopes**,
not probabilities or belief targets. For each information-consistent hidden card type with positive
safe unseen multiplicity, construct only public post-recruit material and call the pure terminal
evaluator. Guaranteed terminal bits require the same exact outcome across the entire support; support
bits require at least one matching outcome. The actual face-down card is never inspected, sampled,
or identified. The model has no belief auxiliary head and the trainer applies no separate loss to
these fields.

## 4. v2 model and fair initialization

### Architecture

`candidate-structured-residual-mlp-v2` has **35,779 trainable parameters**:

```text
base hidden:      Linear(87, 128) -> tanh
base value:       Linear(128, 1) -> base logit
play history:     Linear(314, 32) -> tanh
recruit history:  Linear(314, 32) -> tanh
play consequence: Linear(64, 32) -> tanh
recruit consequence: Linear(54, 32) -> tanh
play residual:    Linear(192, 1)
recruit residual: Linear(192, 1)

logit = base_logit + phase_selected_residual
```

Each residual head consumes the concatenation of the base hidden vector (128), its phase-specific
history representation (32), and its phase-specific consequence representation (32). The play or
recruit residual is selected by the exact v2 phase one-hot; invalid phase encodings are rejected.
The other phase's branch is not used for the output. There is one scalar terminal value logit, no
policy head, no belief head, no opponent model, and no auxiliary objective.

### Initialization

Every v2 fit starts from the same selected q0 tensor used to warm-start Step-2 v1:

* copy q0's v1 hidden weights and bias into `base hidden`;
* copy q0's v1 output weights and bias into `base value`;
* initialize all four structured projection layers with the declared private Xavier-uniform seed;
* initialize both residual-head weight matrices and biases to exactly zero; and
* create a fresh AdamW optimizer with no optimizer-state transfer.

Thus an untrained v2 model is exactly q0's v1 logit on every valid candidate, independent of the
new features. The q0 base trunk and base value are **trainable**, not frozen. Freezing them would
make the architecture comparison include an unbalanced optimization constraint: v1 trains its q0
warm-started trunk, while v2 could not. A frozen-base diagnostic is out of scope and is not a hidden
fifth cell.

The same structured-init seed is used for the q0-only and mixed v2 fits within a replicate; arm
identity never changes initialization. Model construction must pass a q0-initialization equality
fixture before training (logit tolerance `1e-7`, identical greedy choices on the complete fixture
candidate set). The zero residual is intentional even though structured projections receive no
back-propagated gradient until residual weights move away from zero; this is the price of an exact
q0 embedding and is common to both data arms.

## 5. Six v2 fits and training configuration

For each of `C1–C3` and `M1–M3`:

* re-encode the retained Step-2 records with v2, preserving the v1 row order and exact game/pair
  split groups;
* use the same 90/10 complete-game or paired-block split as its retained v1 dataset;
* retain terminal winner and behavior-policy metadata only as audit metadata, never as input;
* use the same split and shuffle seeds recorded in the matching Step-2 plan;
* use the same selected-action terminal MC target and the same dataset sample set;
* use AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size `1024`, maximum 50 epochs,
  patience 8, equal-game-weighted validation BCE checkpoint selection, one Torch CPU thread,
  deterministic algorithms, zero data-loader workers, and no gradient clipping; and
* report pooled/equal-game BCE, Brier, accuracy, calibration, phase metrics, target balance,
  candidate/action/card coverage, examples per second, and wall time.

No class weighting, phase weighting, data-arm weighting, action weighting, oversampling, ranking
loss, TD target, policy loss, belief loss, rollout label, or post-result early stopping rule is
allowed. The v1 checkpoints are immutable controls; they are not silently retrained under v2 code.

### Seed domains

Use a new Step-3 root seed, `2026091203`, and named SHA-256 seed domains. The three replicate IDs
remain `replicate-1`, `replicate-2`, and `replicate-3` so matching is explicit. Derive at least:

* `step3:v2-structured-init:<replicate>`;
* `step3:dataset-reencode:<arm>:<replicate>` (only for deterministic audit materialization);
* `step3:arena:<cell>:<replicate>`; and
* `step3:nested-bootstrap:<statistic>`.

The v2 initialization seed is arm-independent within a replicate. Training shuffle seeds are copied
from the matched Step-2 declaration, not regenerated after seeing results.

## 6. Fresh Step-3 evaluation design

All arena setup seeds are fresh relative to every Step-2 corpus and arena and every earlier retained
run. Within a comparison cell, each setup seed creates the two seat-swapped games of one paired
block, with logical-agent RNG identities preserved across the seat swap. Shared setup blocks are
intentional for matched comparisons; their dependence is retained in the block-level statistics.
No locked-final seed is opened.

All learned candidates, q0 parent, and q1–q4 use the fixed terminal-offense plus terminal-safety
envelope with greedy learned selection (`epsilon = 0`). Canonical heuristic, canonical random, and
historical q0 remain their declared comparator policies without silently adding the envelope. The q0
parent is the selected `q0-terminal-safety-v1` tensor/checkpoint, not historical q0.

### Physical development schedule

| Cell | Cells | Paired blocks per cell | Physical games |
| --- | ---: | ---: | ---: |
| Within-arm v2 versus matched v1 | 6 (C1–C3, M1–M3) | 500 | 6,000 |
| Mixed v2 versus q0-only v2, same replicate | 3 | 500 | 3,000 |
| Each v2 versus fixed q0 parent | 6 | 500 | 6,000 |
| Each v2 versus canonical heuristic | 6 | 300 | 3,600 |
| q0 parent versus heuristic reference | 1 per replicate, shared by both arm candidates | 300 | 1,800 |
| Each v2 versus canonical random | 6 | 200 | 2,400 |
| Each v2 versus historical q0 | 6 | 200 | 2,400 |
| Each v2 versus each enveloped q1, q2, q3, q4 | 24 | 100 | 4,800 |
| **Total** |  | **15,000 paired blocks** | **30,000 games** |

The 500-block architecture domains are aligned across the q0-only and mixed direct cells, so the
architecture-by-data interaction can use matched blocks. The three 300-block q0-parent/heuristic
reference cells are aligned to both corresponding v2-versus-heuristic cells in that replicate. The
q1–q4 cells are stress/descriptive comparisons because those checkpoints are descendants and, for
q1–q4, part of the Step-2 mixed population.

The external opponent macro is predeclared as the equal-weight average of eight v2 win rates:
q0 parent, heuristic, random, historical q0, q1, q2, q3, and q4. It is descriptive/selection
information, not a replacement for the direct architecture statistic.

## 7. Statistical analysis

The paired-block score is candidate wins divided by two, hence it is in `{0, 0.5, 1}`. Report every
cell per replicate, both seats, score margin, turns/decisions, terminal reasons, throughput, and
safety diagnostics.

Use the Step-2 nested bootstrap contract unchanged: 20,000 deterministic resamples, outer resampling
of the three training replicates and inner resampling of matched paired setup blocks, 95% nearest-rank
endpoints at indices 499 and 19,499. Materialize all three outer replicate draws before consuming any
inner block randomness. For pooled architecture and interaction statistics, draw replicate IDs
jointly for both data arms and use the same 500 inner block indexes for the q0-only and mixed
architecture cells within each drawn replicate occurrence. Compute the interaction blockwise as
`mixed(v2-v1) - control(v2-v1)`. Retain and independently revalidate pair IDs, physical seats,
logical-agent RNG identities, and setup seeds across every aligned cell.

For the eight-opponent macro, each drawn replicate uses independently named inner streams per
opponent because block counts differ; compute each opponent mean first and then take the equal-weight
macro. Do not pool games, decisions, models, checkpoints, or opponent cells as independent
observations.

Predeclare these estimands:

1. **Architecture effect by data arm:** v2-v1 paired score in the q0-only arm and in the mixed arm.
2. **Pooled architecture main effect:** equal-arm average of the two arm-specific effects, with
   common within-replicate block resamples.
3. **Data effect under v2:** mixed-v2 minus q0-only-v2 paired score.
4. **Data effect under v1:** retained Step-2 mixed-v1 minus q0-only-v1 result, reported only as a
   descriptive cross-step reference because its arena seed domain differs.
5. **Data x architecture interaction:** `(mixed v2 - mixed v1) - (q0-only v2 - q0-only v1)`;
   the signed positive direction means structured features help more on mixed data. Use aligned
   architecture blocks and the same inner block indexes within a replicate.
6. **External robustness:** v2 versus q0 parent, random, heuristic, historical q0, and q1–q4,
   plus candidate-minus-parent difference on the shared heuristic reference block.
7. **Equal-opponent macro:** the eight-opponent macro above, with its own nested block/replicate
   uncertainty and no substitution for item 1.

### Advancement classification

Each v2 data arm has the following frozen recipe gate:

* v2 versus its matched v1: nested lower endpoint strictly above 50%, and at least two of three
  replicate point estimates above 50%;
* v2 versus q0 parent: nested lower endpoint above 50%;
* v2 versus random: nested lower endpoint above 50%;
* v2-minus-q0-parent heuristic reference: nested lower endpoint above -5 percentage points;
* minimum v2 seat point estimate across every `(replicate, listed cell, physical seat)` row for the
  v2 candidate in the v2-v1, q0-parent, heuristic, and random cells is at least 45%; the losing v1
  side of direct architecture play is never included;
* zero avoidable immediate losses and zero missed guaranteed current-turn wins under the fixed
  envelope; and
* all replay, schedule, seed, split, source, checkpoint, and information-safety checks pass.

A data arm that satisfies every condition is an **advancing structured recipe**. The architecture
has a general advancement claim only if both data arms satisfy the gate and the pooled architecture
main effect is above 50%; if exactly one arm satisfies it, report **data-dependent recipe advance**
and do not claim a universal architecture improvement. If neither satisfies it, report
`does_not_advance` or `inconclusive_does_not_advance` according to whether the primary intervals
exclude or cross 50%; Step 4 remains authorized.

The cross-data, q1–q4, historical-q0, and equal-opponent results cannot be promoted to a primary
gate after seeing the data.

### Predeclared Step-4 recipe selection

Step 4 must have one structured architecture/data recipe before it produces any counterfactual
rollout labels. Select it from the two v2 data arms using only this Step-3 development evidence:

1. reject any arm whose information-safety or replay/tactical invariant fails;
2. for each remaining arm calculate the **robustness floor** as the minimum of these five margins:
   `lower(v2-v1) - 0.50`, `lower(v2-q0) - 0.50`, `lower(v2-random) - 0.50`,
   `lower(v2-minus-parent-heuristic) - (-0.05)`, and `minimum-seat - 0.45`;
3. select the arm with the larger robustness floor;
4. break an exact floor tie by larger equal-opponent macro point estimate, then larger
   v2-versus-v1 point estimate, then the fixed arm order q0-only before mixed; and
5. if both arms fail safety/integrity, do not create a structured Step-4 target; retain q0 as the
   fallback and report the block explicitly.

This rule selects a **development-selected Step-4 input recipe**, not a lucky replicate, advancing
checkpoint, or promoted champion. Step 4 retrains that selected v2 architecture on its selected data
arm under its separately declared counterfactual labeling plan. If v2 does not advance, the selected
recipe still feeds Step 4 under the rule above; no post-result corpus deletion or locked-final
selection is permitted.

## 8. Safety, information, and implementation tests

Before claim execution, the implementation must pass:

* v2 accepts only `PlayerObservation` plus one legal semantic action and imports no authoritative
  `GameState` or transition function;
* hidden-state metamorphic tests: states with identical acting-player observation and legal actions
  but different opponent hand, deck order, or current face-down identity produce identical v2
  vectors, logits, and selected actions;
* prior completed face-down cards may affect history because they are public after completion, while
  the current recruit face-down card cannot affect any feature;
* every feature name is traced to a safe observation field or a pure deterministic consequence of
  public fields; no feature contains a seed, true hidden card, private hand, or realized draw;
* v1-prefix equality, history padding/order golden vectors, play-branch golden vectors, recruit
  hidden-support golden vectors, duplicate-card offers, terminal effects, and all card types;
* candidate tuple/order permutation invariance and semantic action return identity;
* q0-embedded v2 initialization equality and deterministic CPU tensor digest;
* no auxiliary output, belief probability, policy target, or counterfactual action label appears in
  the dataset or model manifest;
* fixed offense/safety envelope audit: zero missed guaranteed wins and zero avoidable guaranteed
  losses for every v2 and v1 candidate in all retained arena records; and
* a source audit rejects accidental calls to authoritative successor simulation from the encoder or
  consequence builder.

## 9. Artifacts, validator, and runtime contract

Write claim artifacts under `runs/m7-structured-model-v2/`:

```text
plan.json                         immutable 2x2 declaration and seed schedule
source-identity.json              Git, lockfile, rules/code fingerprints
inputs/step2-reference.json       six corpus/dataset/v1 checkpoint fingerprints
encoder-v2-schema.json            519 names, scales, definitions, golden digest
datasets/{C,M}{1,2,3}/            v2 arrays, manifest, split/group audit
checkpoints/{C,M}{1,2,3}/         immutable v2 bundle, metrics, tensor digest, lineage
training/{C,M}{1,2,3}/            history, runtime, deterministic settings
arenas/<cell>/                    compressed semantic game records and aggregate report
statistics.json                   cell, main-effect, interaction, macro bootstraps
safety-report.json                hidden-state, envelope, replay, and terminal audits
selection.json                    frozen Step-4 recipe selection and rationale
result.json                       integrity, status, counts, fingerprints, interpretation
checksums.json                    retained artifact checksum manifest
```

The independent validator must reload the plan and source identities, verify the six exact Step-2
inputs, reconstruct every v2 feature through validator-local safe-observation logic without calling
the production v2 encoder, confirm identical split groups/labels, check q0 initialization and all
checkpoint manifests, recompute every arena aggregate from retained records without calling the
production arena/statistics/selection entry points, replay every safety audit, and reproduce nested
bootstrap endpoints and the selection rule. It may share only stable semantic storage, public
observation construction, and the pure engine terminal adjudicator. Preserve the Step-2 repair
lesson: all outer replicate draws are materialized before any inner draws. One validator-only
operational repair is allowed; it cannot change games, seeds, weights, arenas, estimands, thresholds,
or decision rule.

Use one substantial process at a time. The nominal CPU budget is:

| Work | Budget allocation |
| --- | ---: |
| plan, input audit, v2 re-encoding, feature/safety fixtures | 45 min |
| six deterministic v2 trainings | 180 min |
| 30,000 arena games and record verification | 90 min |
| statistics, independent validation, checksum/result artifacts | 90 min |
| reserve | 60 min |
| **total claim cutoff** | **7 h 45 min** |

Before claim dispatch, a non-claim timed smoke must exercise v2 encoding, one q0-embedded training
fit, one arena, and independent validation. Freeze a linear extrapolation showing the six fits,
30,000 arena games, full replay/safety checks, and 20,000-resample validation fit within 7 hours 45
minutes; otherwise reduce no scientific sample and report implementation blocked.

Training remains CPU-only with one Torch thread; no GPU, distributed actor, or concurrent claim job.
A tiny smoke run may precede the claim run but cannot be included in claim statistics. If the hard
eight-hour limit is reached, stop without opening any additional arena or locked-final block and
report an infrastructure-incomplete result.

## 10. Risks and mitigations

* **History leakage:** current versus completed-turn handling could accidentally expose the current
  face-down card. Enforce state-phase fixtures and hidden-state metamorphic tests.
* **Pseudo-belief leakage:** support fractions could be misread as probabilities. Name them support
  envelopes, never train a belief target, and validate they use only safe unseen counts.
* **Architecture confounding:** starting v2 from a Step-2 v1 checkpoint or freezing the q0 trunk
  would make the comparison asymmetric. Start every v2 from q0, train the base, and preserve the
  exact q0 zero-residual embedding.
* **Branch imbalance:** recruit has two actions and play has up to twelve. Report phase coverage and
  equal-game validation; do not reweight phases after seeing results.
* **Step-2 artifact compatibility:** adding the v2 module changes the package code fingerprint.
  Preserve the original Step-2 fingerprints and require explicit historical replay/rules
  compatibility rather than silently bypassing verification.
* **Shared arena blocks:** the same q0 reference is intentionally shared by both data arms in the
  heuristic comparison. Record the dependency and use matched block bootstrap, never treat those
  cells as independent games.
* **Descendant stress cells:** q1–q4 are correlated with Step-2 data and are descriptive only.
* **Runtime:** 30,000 games plus six fits is bounded by the claim budget; resumable per-cell arena
  records and a pre-materialized plan permit restart without changing identity.
* **Small replicate count:** three replicates are retained as the independent training unit; no
  model-level pooling or extra replicate may be added after looking at results.

**Interpretation limit:** a positive result would support this combined encoder-plus-architecture
recipe under the fixed selected-action MC target and evaluation envelope. It would not isolate
history from consequence features or residual phase routing, prove counterfactual action quality,
correct beliefs, optimal play, or a general architecture advantage outside these two retained data
arms.

## Frozen implementation and smoke

- `b1f2b6f`: 519-feature encoder, 35,779-parameter residual model, aligned dataset/trainer/checkpoint,
  and structured inference agent.
- `2bfe619`, `2e18f15`, `a74d21c`: resumable experiment, explicit claim CLI, deadline/checksum/result
  boundaries, independent 519-feature validator, and phase timing.
- `68ed5fe`, `388a001`, `e6d2327`, `7dcae27`: full Step-2/archive compatibility enforcement,
  historical allowlist, local complete arena reconstruction, record-level alignment, and expected
  policy config reconstruction.
- `7321aa1`: 22 adversarial validator tests covering report, schedule, seat, policy, and RNG
  tampering.

A bounded retained-input smoke at `/tmp/agent-avenue-step3-smoke-v3` completed six one-epoch fits, 60
one-pair arenas, production statistics/selection, and independent validation. Plan fingerprint:
`191fbd969f9adabc1506e517e16941775c651b049097fc8ab471175d0cfd0d6a`; result fingerprint:
`fb49c1692320088a6abd83ebd47e59c300610b7c2db076f2c9b6018bbe107302`; validation fingerprint:
`f5efbdbf4f978b48a5a4fa5f58e0a0dc455b9ef856a6e3b169e6290bd851fcde`.

Observed phase timings project the full runner at 210.15 minutes and independent validator at 111.86
minutes, 322.02 minutes combined. This is below the 7-hour-45-minute claim cutoff without reducing
any frozen sample. Ruff, strict mypy, and 290 tests pass. The final fresh preflight review is GO.
