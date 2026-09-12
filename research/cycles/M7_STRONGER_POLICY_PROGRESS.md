# Stronger policy program: progress and decision log

**Program:** `m7-stronger-policy-program-v1`
**Last updated:** September 12, 2026
**Overall state:** Step 2 complete; Step 3 exact agreement being finalized
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
| 3. Structured model v2 | Designing | [`M7_STRUCTURED_MODEL_V2.md`](M7_STRUCTURED_MODEL_V2.md) | Pending | Pending | Pending |
| 4. Counterfactual rollouts | Authorized, waiting | Pending | Pending | Pending | Pending |
| 5. Independent league | Authorized, waiting | Pending | Pending | Pending | Pending |

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

## Current step: structured model v2 design

Step 3 is a predeclared 2x2 data-by-architecture study. The retained Step-2 v1 checkpoints are
controls; six new structured v2 models train on the same q0-only and mixed datasets. The proposed v2
keeps the 87-feature prefix and adds completed public history plus safe play/recruit consequence
blocks, with separate zero-initialized residual heads over the q0-initialized trainable base. It keeps
the selected-action terminal MC objective and fixed tactical envelope; belief heads and
counterfactual rollout labels remain Step 4.

Fresh Step-3 arenas will rerun both v1 and v2 controls on new disjoint blocks, estimate architecture
effects within both data arms and their interaction, then select one structured recipe for Step 4 by
a frozen robustness-floor rule even if v2 does not advance.

## Completed implementation work

- `de1872a`: explicit agent RNG identities with backward-compatible default schedules.
- `b99cc2d`: terminal-offense confirmation runner, independent public oracle, tie diagnostics, and
  validator foundation.
- `b18a13e`: exhaustive setup holdout and corrected guaranteed-win prefix endpoints.
- `23fe496`: independent validator path, frozen provenance checks, whole-step budget metadata, and
  descriptive-only direct comparison.
- Current validation: Ruff and strict mypy pass; 250 tests pass.
- Step-2 implementation: deterministic assignment/schedules, resumable six-corpus/six-training
  pipeline, 54 development arenas, exhaustive holdout/source/deadline checks, nested bootstrap,
  decision logic, and a separate independent validator are committed and smoke-tested.

## Decision log

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
