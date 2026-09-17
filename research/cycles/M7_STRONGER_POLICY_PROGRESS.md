# Stronger policy program: progress and decision log

**Program:** `m7-stronger-policy-program-v1`
**Last updated:** September 17, 2026
**Overall state:** Step 5 exact design frozen; implementation and preflight pending
**Current selected policy:** `q0-terminal-safety-v1`

## Goal snapshot

Produce a stronger information-safe policy by proceeding in this fixed order:

1. terminal-offense confirmation and RNG hardening;
2. population replay with encoder/model v1;
3. structured public-history/candidate-consequence model v2;
4. information-safe counterfactual rollout supervision; and
5. independent league evaluation and promotion.

The main suspected failure is not insufficient parameter count. Current training labels only the
selected action with the eventual winner, while inference maximizes over all legal actions. The model
therefore extrapolates unsupported alternatives, must rediscover exact rules statistically, omits
public-history signaling, and specializes to narrow self-play distributions.

## Evidence entering the program

- q0 strength audit: 72,000 games and 1,015,338 decisions, independently reproduced.
- q0 missed 589/1,832 publicly guaranteed immediate wins and zero avoidable provable immediate
  losses.
- q0 training corpus retrospective: 1,473 guaranteed-win opportunities; exploratory behavior missed
  154, of which 118 still ended in wins. Those selected nonlethal actions received positive Monte
  Carlo targets while the unchosen guaranteed actions received no same-position labels.
- Diagnostic immediate-win wrapper improved q0 by 0.25–1.60 percentage points across the declared
  suite, but distinct policy RNG identities remained a caveat for the smallest effects.
- q1–q4 strongly exploited q0, and the heuristic-ranking warm start did not improve its paired MC
  controls.

## Step table

| Step | State | Exact agreement | Implementation | Claim run | Result |
| --- | --- | --- | --- | --- | --- |
| 1. Terminal offense/RNG | Complete | [`M7_TERMINAL_OFFENSE_CONFIRM_V1.md`](M7_TERMINAL_OFFENSE_CONFIRM_V1.md) | `de1872a`, `b99cc2d`, `b18a13e`, `23fe496` | 60,000 games; validated | [`docs/M7_TERMINAL_OFFENSE_CONFIRM_RESULTS.md`](../../docs/M7_TERMINAL_OFFENSE_CONFIRM_RESULTS.md) |
| 2. Population replay v1 | Complete; does not advance | [`M7_POPULATION_REPLAY_V1.md`](M7_POPULATION_REPLAY_V1.md) | `027b434`, `fb696f8`, `adc55a2`, `bf7cb09`, `c9b78c3` | 24,000 training + 24,000 development; repaired validation passed | [`docs/M7_POPULATION_REPLAY_RESULTS.md`](../../docs/M7_POPULATION_REPLAY_RESULTS.md) |
| 3. Structured model v2 | Complete; does not advance | [`M7_STRUCTURED_MODEL_V2.md`](M7_STRUCTURED_MODEL_V2.md) | `b1f2b6f` through `7321aa1` | Six fits + 30,000 games; exact-source validation passed after one operational repair | [`docs/M7_STRUCTURED_MODEL_V2_RESULTS.md`](../../docs/M7_STRUCTURED_MODEL_V2_RESULTS.md) |
| 4. Counterfactual rollouts | Complete; inconclusive, does not advance | [`M7_COUNTERFACTUAL_ROLLOUT_SUPERVISION_V1.md`](M7_COUNTERFACTUAL_ROLLOUT_SUPERVISION_V1.md) | `b20699b` through `cb0e045` | 24,000 games; repaired exact-source validation passed | [`docs/M7_COUNTERFACTUAL_ROLLOUT_RESULTS.md`](../../docs/M7_COUNTERFACTUAL_ROLLOUT_RESULTS.md) |
| 5. Independent league | Exact design frozen; implementation pending | [`M7_INDEPENDENT_LEAGUE_PROMOTION_V1.md`](M7_INDEPENDENT_LEAGUE_PROMOTION_V1.md) | None | None | Sole promotion challenger is q0 terminal offense; M-v2 descriptive |

## Completed step 1

The 60,000-game confirmation and independent validator completed in 5 hours 9 minutes with no repair.
Every structural criterion passed. Treatment converted 3,385/3,385 guaranteed wins, exact q0-family
maximum-logit ties were zero in 455,629 decisions, and 28,000 matched control/treatment games had zero
pre-endpoint RNG/action-prefix failures. The anchor macro improved by +0.75 percentage points with a
95% interval of +0.60 to +0.908, passing the separate practical criterion. The tactical envelope is
now fixed for steps 2–4; no checkpoint or web default changed.

## Completed step 2

Mixed-population model-v1 collection beat matched q0-only retraining directly in every replicate and
62.10% overall [60.13%, 64.00%]. It improved matched heuristic performance by +6.33 percentage
points and q1–q4 stress performance by roughly +12 to +14.5 points. It formally does not advance:
the treatment-minus-parent heuristic lower endpoint was -5.056 points against a strict greater-than
-5 threshold, and the all-candidate seat rule was triggered by the losing control side of the direct
arena. Treatment's own lowest observed seat point was 54.0%.

The claim runner completed at `fde19b5`. One validator-only repair corrected nested-bootstrap RNG
consumption order; repaired validation at `2bf8822` reproduced all evidence and the unchanged
decision. Both data arms, six datasets, and six v1 checkpoints remain fixed inputs for Step 3.

## Completed step 3

The combined structured v2 recipe beat matched v1 controls in both arms: q0-only 53.93%
[51.47%, 56.63%], mixed 55.80% [52.73%, 58.67%], and pooled 54.87% [52.50%, 56.85%]. Mixed-v2
also beat q0-only-v2 at 64.63% [59.07%, 69.07%]. The interaction was inconclusive, so there is no
claim that mixed replay benefits disproportionately from v2.

Both arm gates failed only the parent-heuristic non-regression condition. Mixed was closer at -3.56
points with interval [-8.83, +1.67], versus q0-only -8.0 [-12.0, -4.22]. All candidate seat,
q0-parent, random, tactical, and integrity gates passed. The frozen robustness-floor rule selects M
as a development input for Step 4, not as an advancement or promotion.

One operational repair restored wrapper-managed `driver.stdout` to the empty hash already captured
by checksums; no scientific artifact/source changed. Exact-source independent validation then
passed. Future checksum scopes exclude wrapper logs. Combined claim plus validation took about 3h36.

## Reauthorized step 4 repairs

On September 17, repair 2 passed. It reconstructed the exact retained plan from source `576b896`,
proved byte-exact equality for all 1,400 replicate-1 rows, generated the two remaining replicates,
and published three canonical 280-position targets with finite integer-equivalent one-ULP metadata.
The fresh-context review disposition was GO.

The authorization pause left attempt 1's wall-clock cutoff stale even though only 615.565 seconds of
active claim phases had run. Repair 3, the final trivial repair allowed for Step 4, preserved that
consumed active time, excluded only the inactive pause, and resumed the original runner from exact
source using canonical path serialization. It changed no scientific artifact or rule.

On September 13 the user authorized more than one strictly trivial repair. The new policy allows up
to three mechanically proven, estimand-preserving repairs per step while continuing to prohibit any
change to models, targets, seeds, samples, arenas, statistics, or gates.

Step-4 repair 2 reused exact replicate-1 rows, completed one-position packaging for all replicates,
and validated terminal/leaf metadata through integer counts plus a one-ULP bound. Repair 3 then
completed the original runner and validator. The previously committed failure report remains the
record of the first authorization boundary.

## Completed step 4

Rollout supervision beat matched structured-v2 controls at 53.03% [51.80%, 54.30%], with all three
direct replicate points above 50%. It passed q0-parent, random, seat, tactical, and integrity gates.
It failed only the aligned heuristic non-regression gate: treatment minus the q0-parent heuristic
reference was -2.06 points with interval [-6.11, +2.28], whose lower endpoint did not exceed -5.
The frozen classification is `inconclusive_does_not_advance`.

Three mechanically scoped repairs preserved the exact plan and scientific payloads. Runner and
independent validator exited zero, and checksum reconstruction found 1,953/1,953 exact payloads.
The rollout treatment does not enter Step 5; retained mixed-data structured-v2 M is the descriptive
entry. The post-run validator target rate of 64.45 units/s is retained as a runtime caveat, not a
retroactive gate: binding preclaim eligibility passed and completed runtime stayed below cutoff.

## Frozen step 5

The final locked league freezes twelve policies, two disjoint 200-setup families, all 66 unordered
matchups, 52,800 physical games, and independent replay validation. `q0-terminal-offense-v1` is the
sole promotion-eligible challenger to `q0-terminal-safety-v1`. q1-q4, M1-M3, historical q0,
heuristic, and random are descriptive only. M1-M3 remain an exchangeable family with no replicate
selection.

Promotion requires the aligned three-anchor tactical lift lower endpoint to exceed +0.25 percentage
points plus exact no-worse, tactical, seat, random, provenance, holdout, and validation gates. The
input registry fingerprint is `df238b7d9c948563ee15e4e2cfc59fe1525a9721b10e24362a7d251c1dfb7d48`.
No locked-final game has been run.

## Completed implementation work

- `de1872a`: explicit agent RNG identities with backward-compatible default schedules.
- `b99cc2d`: terminal-offense confirmation runner, independent public oracle, tie diagnostics, and
  validator foundation.
- `b18a13e`: exhaustive setup holdout and corrected guaranteed-win prefix endpoints.
- `23fe496`: independent validator path, frozen provenance checks, whole-step budget metadata, and
  descriptive-only direct comparison.
- Current validation: Ruff and strict mypy pass; 315 tests pass.
- Step-4 implementation: safe latent rollout core, target artifacts, paired auxiliary trainer,
  control reproduction, 54-cell experiment, complete independent target/arena/statistics/tactical
  validator, and result/checksum/deadline boundaries are committed.
- Step-4 smoke: one position in every stratum, exact controls, targets, treatment fits, arena, and
  independent validation passed. Production and validator target rates are about 160 units/s;
  corrected combined projection is 204.73 minutes with leaf fractions below 50% and zero cap errors.
- Step-3 implementation: structured encoder/model/data/checkpoint/agent, six-fit runner, 60-cell
  evaluation, nested interaction statistics, robustness-floor selection, historical compatibility,
  deadline/resume/checksum boundaries, and independent local feature/arena/statistics validation are
  committed.
- Step-3 smoke: six one-epoch fits plus 120 games and independent validation passed; full runner plus
  validator projection is 322.02 minutes, below the 7h45 claim cutoff.
- Step-2 implementation: deterministic assignment/schedules, resumable six-corpus/six-training
  pipeline, 54 development arenas, exhaustive holdout/source/deadline checks, nested bootstrap,
  decision logic, and a separate independent validator are committed and smoke-tested.

## Decision log

- **September 17 — Step-5 design freeze:** twelve-policy two-family locked league, 52,800 games,
  q0 terminal offense as sole promotion challenger, descriptive M replicate family, aligned anchor
  bootstrap, exact tactical gates, and independent validation are fixed before implementation.
- **September 17 — Step-4 result:** rollout supervision beat matched controls at 53.03% [51.80%,
  54.30%] but failed the heuristic non-regression lower bound; classify
  `inconclusive_does_not_advance`. Retain Step-3 M-v2 as the descriptive Step-5 entry.
- **September 17 — repair 2 complete / repair 3 authorized:** all three frozen rollout targets were
  published after exact-row and integer/one-ULP validation. The final trivial repair preserves the
  615.565 seconds of active attempt-1 budget while excluding the multi-day inactive authorization
  pause and resumes the exact-source runner with canonical plan-path serialization.
- **September 13 — repair policy v2:** user authorized up to three strictly mechanical,
  scientifically invariant repairs per step. Step-4 trivial repair 2 is approved to accept integer-
  equivalent one-ULP terminal metadata and resume the original claim.
- **September 12 — Step-4 stop:** attempt 1 failed at shard assembly. The only repair reproduced all
  replicate-1 rows exactly but refused a staged target over a one-ULP terminal-fraction comparison.
  No scientific result exists, the repair allowance is exhausted, and Step 5 is blocked.
- **September 12 — step-4 implementation freeze:** rollout core and experiment workers completed
  the provenance-safe latent state, 10-world depth-nine target teacher, exact control reproduction,
  paired rollout-loss trainer, claim evaluation, and fully independent numeric target/arena/checksum
  validation. Final preflight is GO at a 204.73-minute combined projection.
- **September 12 — step-4 agreement freeze:** 280 positions/replicate across 14 feasible strata,
  10 copy-weighted worlds/candidate, depth-nine q0-leaf targets, safe-only target identity, independent
  policy permutations/per-decision RNG, fixed 0.20 rollout BCE, exact control reproduction, and a
  hard measured-throughput gate before the 24,000-game claim.
- **September 12 — step-3 result:** structured v2 beat v1 in both data arms, but both failed only the
  frozen parent-heuristic non-regression gate. Select mixed-v2 as the development Step-4 input by the
  predeclared robustness floor; no policy is promoted.
- **September 12 — step-3 repair:** exact-source validation passed after restoring mutable
  `driver.stdout` to its already-checksummed empty state. Future checksum code excludes wrapper logs.
- **September 12 — step-4 design review:** do not freeze the initial 520,000-full-rollout proposal
  until measured rollout and independent-recomputation throughput demonstrate an eight-hour fit.
- **September 12 — step-3 implementation freeze:** the structured core, six-fit/60-cell experiment,
  independent validator, full historical input compatibility, and 22 adversarial validator mutation
  tests are committed. A retained-input smoke passed and projects 5h22 combined execution.
- **September 12 — step-3 agreement freeze:** 519-feature safe encoder, 35,779-parameter q0-embedded
  residual model, six fits across both Step-2 data arms, 30,000 fresh games, exact aligned interaction
  bootstrap, independent feature/statistics validation, and a frozen Step-4 robustness-floor
  selection rule. Full Step-2 input/file/checkpoint identities are committed separately.
- **September 12 — step-2 result:** mixed replay strongly beat matched q0-only controls and improved
  descendant/heuristic stress results but does not advance under the frozen parent-heuristic and
  all-candidate seat gates. Preserve both data arms for the 2x2 Step-3 study.
- **September 12 — step-2 repair:** repaired validation passed after the one authorized
  validator-only RNG-order correction; claim artifacts and decision remained unchanged.
- **September 12 — step-3 design:** use a q0-embedded structured residual model with an unchanged v1
  prefix, eight completed public turns, safe play/recruit consequence blocks, and no belief or
  counterfactual targets. Train on both fixed data arms and evaluate on fresh blocks.
- **September 11 — step-2 implementation freeze:** schedule and experiment workers completed the
  population assignment, six-model training/evaluation pipeline, independent validator, and three
  bounded preflight hardening passes. The final clean source has no unresolved reviewer blocker.
- **September 11 — step-2 pre-claim correction:** the listed candidate comparisons total 7,400
  games per replicate; the separately required 300-pair parent heuristic reference adds 600, so the
  frozen physical total is 8,000 per replicate / 24,000 overall. The reference now shares the exact
  candidate heuristic setup block.
- **September 11 — step-2 freeze:** three paired 4,000-game corpus replicates per arm, exact mixed
  population weights, matched q0 initialization/training, 22,200 candidate-comparison games plus
  1,800 matched parent-reference games, nested replicate/block uncertainty, and no post-result corpus
  selection for step 3.
- **September 11 — step-1 result:** all structural criteria and the practical-lift criterion passed.
  Retain the terminal-offense envelope as a fixed component for steps 2–4. The direct common-RNG
  result remains descriptive, and no selected champion or web default changes before step 5.
- **September 11 — step-2 design direction:** collect fresh paired q0-only and mixed-population
  corpora rather than training on prior evaluation records; use three corpus/training replicates and
  nested block/replicate uncertainty while preserving the model-v1 recipe.
- **September 11 — step-1 freeze:** use 60,000 fresh games with aligned RNG identities, exhaustive
  prior setup exclusion, independent oracle/tie/prefix validation, and a +0.25 percentage-point
  anchor-macro lower-bound threshold for the separate practical-lift claim. Structural correctness
  does not depend on lift significance.
- **September 11 — subagent execution:** one worker implemented the RNG seam; a second implemented
  the confirmation machinery and two bounded preflight hardening passes; a separate fresh reviewer
  identified and resolved agreement, validation-independence, holdout-scope, and direct-RNG
  interpretation issues before claim dispatch.
- **September 11 — user approval:** execute all five steps in order, use subagents with Shelley as
  orchestrator, allow up to eight compute hours per step, and preserve goal/plan/progress notes.
- **September 11 — program boundary:** no champion or web default changes before step 5. The exact
  tactical envelope may be used as a fixed component in steps 2–4 after step-1 structural
  validation because a publicly guaranteed current-turn win terminates in victory by definition.

## Compaction recovery instructions

On context loss, read in order:

1. `AGENTS.md` and `README.md`;
2. `research/cycles/M7_STRONGER_POLICY_PROGRAM_V1.md`;
3. this progress file;
4. the exact agreement/result for the current step;
5. `docs/Q0_STRENGTH_AUDIT_RESULTS.md`; and
6. current `git status`, recent commits, and retained runtime state under
   `runs/research-cycles/m7-stronger-policy-program-v1/`.

Do not start a claim-generating command unless the current step's exact agreement and source revision
are committed. Do not run more than one substantial job concurrently.
