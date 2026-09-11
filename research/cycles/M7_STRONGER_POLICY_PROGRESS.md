# Stronger policy program: progress and decision log

**Program:** `m7-stronger-policy-program-v1`
**Last updated:** September 11, 2026
**Overall state:** Step 1 complete; Step 2 agreement frozen and implementation pending
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
| 2. Population replay v1 | Agreement frozen | [`M7_POPULATION_REPLAY_V1.md`](M7_POPULATION_REPLAY_V1.md) | Pending | Pending | Pending |
| 3. Structured model v2 | Authorized, waiting | Pending | Pending | Pending | Pending |
| 4. Counterfactual rollouts | Authorized, waiting | Pending | Pending | Pending | Pending |
| 5. Independent league | Authorized, waiting | Pending | Pending | Pending | Pending |

## Completed step 1

The 60,000-game confirmation and independent validator completed in 5 hours 9 minutes with no repair.
Every structural criterion passed. Treatment converted 3,385/3,385 guaranteed wins, exact q0-family
maximum-logit ties were zero in 455,629 decisions, and 28,000 matched control/treatment games had zero
pre-endpoint RNG/action-prefix failures. The anchor macro improved by +0.75 percentage points with a
95% interval of +0.60 to +0.908, passing the separate practical criterion. The tactical envelope is
now fixed for steps 2–4; no checkpoint or web default changed.

## Current step: population replay design

Step 2 is frozen as three paired corpus/training replicates. Each replicate compares 4,000 q0-only
control games with 4,000 size-matched mixed-population games using exact 40% q0 and 10% each
q1–q4/heuristic/random logical-slot marginals. The model-v1 encoder, architecture, selected-action MC
loss, optimizer, q0 initialization, epsilon 1/5, and adopted tactical envelope remain fixed.

The run will produce 24,000 training games and 22,200 fresh development games. Nested uncertainty
resamples both training replicates and paired setup blocks. Both control and population datasets feed
step 3 regardless of the step-2 advancement decision, enabling a predeclared data×architecture
comparison rather than post-result corpus selection.

## Completed implementation work

- `de1872a`: explicit agent RNG identities with backward-compatible default schedules.
- `b99cc2d`: terminal-offense confirmation runner, independent public oracle, tie diagnostics, and
  validator foundation.
- `b18a13e`: exhaustive setup holdout and corrected guaranteed-win prefix endpoints.
- `23fe496`: independent validator path, frozen provenance checks, whole-step budget metadata, and
  descriptive-only direct comparison.
- Current validation: Ruff and strict mypy pass; 223 tests pass.

## Decision log

- **September 11 — step-2 freeze:** three paired 4,000-game corpus replicates per arm, exact mixed
  population weights, matched q0 initialization/training, 22,200 fresh development games, nested
  replicate/block uncertainty, and no post-result corpus selection for step 3.
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
