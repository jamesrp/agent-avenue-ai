# Stronger-policy program v1: Step-4 briefing

**Program:** `m7-stronger-policy-program-v1`
**Status:** Step 5 design-frozen; implementation and preflight pending
**Updated:** October 7, 2026
**Agreement:** [`research/cycles/M7_STRONGER_POLICY_PROGRAM_V1.md`](../research/cycles/M7_STRONGER_POLICY_PROGRAM_V1.md)

## Executive summary

The program has completed four of five authorized steps:

1. The exact immediate-win envelope is structurally correct and measurably improves q0.
2. Mixed replay and structured v2 each produced large direct gains over matched controls.
3. Both earlier learned recipes failed the frozen q0-parent heuristic non-regression gate.
4. Counterfactual rollout supervision produced a statistically positive matched-recipe gain, but it
   also failed that same robustness gate and does not advance.
5. Step 5 will independently evaluate the retained descriptive entry before any promotion decision.

The selected champion and web default remain `q0-terminal-safety-v1` until Step 5 completes.

## Step outcomes

| Step | Outcome | Main result |
| --- | --- | --- |
| 1. Terminal offense | Adopted for research | +0.75 pp anchor macro [0.60, 0.908]; 3,385/3,385 guaranteed wins converted |
| 2. Population replay v1 | Does not advance | Mixed beat q0-only control 62.10% [60.13%, 64.00%], but robustness gates failed |
| 3. Structured model v2 | Does not advance | V2 beat v1 in both data arms; mixed-v2 selected only as Step-4 development input |
| 4. Counterfactual rollout | Inconclusive; does not advance | Treatment beat matched control 53.03% [51.80%, 54.30%], but heuristic lower bound was -6.11 pp |
| 5. Independent league | Design frozen; implementation pending | Retained Step-3 mixed structured-v2 M is descriptive; q0 terminal offense is the sole promotion challenger |

## Step-4 interpretation

The rollout auxiliary worked in its narrow matched comparison. Direct replicate points were 52.6%,
53.7%, and 52.8%; the nested treatment-control lower endpoint was above 50%. Treatment also beat q0
at 84.17% [81.13%, 88.07%] and random at 87.25% [84.83%, 89.75%]. Its lowest seat point was 47.6%,
and all tactical and integrity checks passed.

It did not satisfy the frozen cross-opponent robustness rule. Relative to q0 against the aligned
heuristic blocks, treatment was -2.06 percentage points with interval [-6.11, +2.28], missing the
strict lower endpoint above -5. The correct classification is `inconclusive_does_not_advance`.
This is evidence of modest matched-recipe improvement alongside continued non-transitivity, not a
claim that rollout supervision is generally harmful.

Three trivial repairs were required to complete Step 4. Fresh scientific and artifact reviews found
them mechanically scoped: they changed shard packaging, bounded derived-summary comparison, and
active-time resume accounting, while preserving every target row, tensor, seed, game, statistic,
threshold, and gate. Exact-source runner and independent validator both passed.

## Current policy status

- Selected hybrid champion: `q0-terminal-safety-v1`.
- Permanent pure-neural baseline: historical q0.
- Terminal offense: validated fixed tactical component for research candidates.
- Mixed structured v2 M: descriptive Step-5 entry, not currently promoted.
- Rollout-supervised T1-T3: valid research checkpoints that do not enter Step 5.

## Step-5 boundary

The exact Step-5 agreement is now frozen. It declares a twelve-policy, two-family locked league with
52,800 physical games. `q0-terminal-offense-v1` is the sole promotion-eligible challenger to the
selected `q0-terminal-safety-v1`; retained M1-M3 are descriptive and cannot be selected or promoted.

Implementation, isolated smoke, source freeze, and fresh preclaim review must complete before any
locked-final game. No additional recipe tuning is authorized.
