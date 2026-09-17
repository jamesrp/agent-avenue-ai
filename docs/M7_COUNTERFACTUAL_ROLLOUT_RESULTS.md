# Counterfactual rollout supervision: completed result

**Program:** `m7-stronger-policy-program-v1`
**Step:** 4 of 5
**Cycle:** `m7-counterfactual-rollout-supervision-v1`
**Status:** Complete; independently validated; `inconclusive_does_not_advance`
**Completed:** September 17, 2026
**Agreement:** [`research/cycles/M7_COUNTERFACTUAL_ROLLOUT_SUPERVISION_V1.md`](../research/cycles/M7_COUNTERFACTUAL_ROLLOUT_SUPERVISION_V1.md)
**Claim source:** `576b896b305273eb29701521d38791712b160a95`

## Decision

The fixed counterfactual-rollout auxiliary improved the matched structured-v2 recipe directly, but
it does **not** advance. Treatment beat exact matched controls at **53.03%**, with nested 95% interval
**[51.8%, 54.3%]**, and all three direct replicate points exceeded 50%. The frozen heuristic
non-regression gate failed: treatment minus the q0-parent heuristic reference was **-2.06 percentage
points**, interval **[-6.11, +2.28]**, against a strict lower endpoint above -5 points.

The rollout-supervised T1-T3 family therefore does not enter Step 5. The retained Step-3 mixed-data
structured-v2 M recipe is the descriptive Step-5 entry. No champion or web default changes in Step 4.

## Frozen experiment

The completed claim retained:

- three 280-position train-only public panels across 14 fixed strata;
- 4,198 legal candidate rows and 41,980 information-consistent hidden-world rollouts;
- ten copy-weighted worlds per candidate, depth nine, and the frozen q0 leaf evaluator;
- three exact matched-control reproductions and three rollout-auxiliary treatments;
- 24,000 fresh development games across 54 arena cells; and
- independent local reconstruction of panels, targets, controls, arenas, statistics, selection, and
  checksum scope.

Control tensor digests reproduced the retained Step-3 M checkpoints byte-for-byte before treatment
results were used. The only treatment difference was the fixed position-balanced rollout BCE term at
weight 0.20.

## Main results

| Estimand | Point | Nested 95% interval |
| --- | ---: | ---: |
| Treatment vs matched control | 53.03% | [51.80%, 54.30%] |
| Treatment vs q0 parent | 84.17% | [81.13%, 88.07%] |
| Treatment vs random | 87.25% | [84.83%, 89.75%] |
| Treatment minus q0-parent vs heuristic | -2.06 pp | [-6.11, +2.28] pp |
| Control minus q0-parent vs heuristic | -4.06 pp | [-8.22, +0.17] pp |

Direct treatment-vs-control replicate points were **52.6%**, **53.7%**, and **52.8%**. The equal-cell
external macro was 77.89% for treatment versus 76.55% for control, a descriptive +1.35-point gain.
The lowest treatment physical-seat point was 47.6%.

Every treatment arena record passed the fixed tactical envelope: zero executed avoidable provable
losses and zero missed publicly guaranteed current-turn wins. All source, split, panel, sampler,
rollout, schedule, checksum, and validator integrity conditions passed.

## Why it did not advance

Seven of eight frozen conditions passed. The sole failure was:

```text
heuristic-aligned lower endpoint = -6.11 percentage points
required lower endpoint          > -5.00 percentage points
```

Because the direct lower endpoint was above 50% but a robustness gate failed, the predeclared
classification is `inconclusive_does_not_advance`, not evidence that rollout supervision is harmful.
The result supports a modest matched-recipe gain and continued non-transitive weakness against the
heuristic reference.

## Repairs and validation

Attempt 1 stopped at a shard-granularity mismatch. Repair 1 proved all 1,400 retained replicate-1
rows byte-exact but stopped on a one-ULP derived metadata comparison. Under the approved trivial-
repair policy, repair 2 accepted only finite integer-equivalent one-ULP summaries and published all
three canonical targets. Repair 3 preserved exactly 615.5649927302729 seconds of consumed active
budget while excluding the inactive authorization pause, then resumed the original exact-source
runner. None changed a model tensor, target row, seed, sample, game, statistic, threshold, or gate.

The exact-source runner and independent validator both exited zero. Validation artifact
`af9116548a2d907895a71e8f2200c751d8f099c8a59922a570c74244e2398acd` passed all eight declared
checks. Independent checksum recalculation found 1,953 expected and actual payload files with zero
missing, extra, or mismatched entries.

The post-run runtime diagnostic measured independent target recomputation at 64.45 units/s, below
the 67-unit prefreeze threshold. This does not retroactively change claim eligibility: the binding
preclaim artifact passed at 159.93 production and 160.32 validator units/s, and completed combined
runtime projected 151.27 minutes, below the 465-minute cutoff. The slower post-run measurement is
retained as a reproducibility caveat.

## Evidence identities

| Item | Identity |
| --- | --- |
| Plan | `d9522b85e3112756710a8d3e5ddd3729f35fe7ba998b586fb10c9a17878b0625` |
| Input manifest | `53ba97aed7c40a87b25f5adc1b7925e28f4aad49bd0bfb758e78b125b54181e8` |
| Input audit | `5591b0a4da966b75709cc0b63ac3b10ec09cc69551c1933be5dfd08c416e5abd` |
| Statistics | `1ec36abb7ca4486c6b2afaebfae21259572e23524f7ad8a6ceed2744ca78bcfd` |
| Selection | `e67d921c615e82a043550512b6f1334b2f7aa3ab7e26fcd7b00a67dbc1710c3c` |
| Result | `7451af95c76cc152ce60d6b0da2845c2df390cafa1404560c3b2e12cf3d60769` |
| Checksums | `18254ba6f5ca13f07ed2a7083a6c5426faa2ae0f491406b078e16da05dfb3444` |
| Fresh review | `c1fa6de564c5882b87d9001f932b3425e63917f1f68062ba68be7df8107326c5` |
| Archive | `90997540e5bf6e642ff72aeeda9261dbd5391f9aeca1a0ec7f0743c5c2377c42` |

Retained evidence is under `runs/m7-counterfactual-rollout-supervision-v1/`. The verified archive is
`artifacts/archive/m7-counterfactual-rollout-supervision-v1-2026-09-17.tar.gz` (57,385,898 bytes),
containing 3,020 members. The historical initial failure remains documented in
[`M7_COUNTERFACTUAL_ROLLOUT_FAILURE.md`](M7_COUNTERFACTUAL_ROLLOUT_FAILURE.md).
