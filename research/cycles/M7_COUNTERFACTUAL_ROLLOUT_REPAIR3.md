# Approved trivial repair 3: Step-4 active-time resume accounting

**Program:** `m7-stronger-policy-program-v1`
**Step:** 4 of 5
**Cycle:** `m7-counterfactual-rollout-supervision-v1`
**Repair:** `trivial-repair-3`
**Status:** Completed; exact-source runner and independent validator passed
**Approved:** September 17, 2026
**Completed:** September 17, 2026
**Original claim source:** `576b896b305273eb29701521d38791712b160a95`
**Repair policy:** [`M7_TRIVIAL_REPAIR_POLICY_V2.md`](M7_TRIVIAL_REPAIR_POLICY_V2.md)

## Trigger

Repair 2 successfully published all three frozen rollout-target artifacts. The retained execution
state still measures the claim cutoff from attempt 1's wall-clock start on September 12. The
multi-day authorization and review pause therefore consumes the cutoff even though no claim process
was running. In addition, executing the exact source from a detached worktree resolves linked input
files to the canonical workspace but serializes some audit paths as absolute rather than the
repository-relative strings retained in the immutable plan.

Neither issue concerns a policy, target, tensor, seed, game, sample, statistic, or gate.

## Frozen scientific plan

Do not alter plan `d9522b85e3112756710a8d3e5ddd3729f35fe7ba998b586fb10c9a17878b0625`,
input audit `5591b0a4da966b75709cc0b63ac3b10ec09cc69551c1933be5dfd08c416e5abd`,
canonical targets, panels, training, arenas, statistics, thresholds, or selection rules.

## Repair scope

1. Authenticate the incomplete one-attempt execution state and exact phase timings.
2. Preserve every attempt record and phase timing. Set only `started_at_epoch_seconds` to the current
   dispatch time minus the already consumed active phase total, then recompute the execution-state
   artifact fingerprint. This preserves approximately 615.565 seconds of consumed claim budget and
   excludes only the inactive authorization pause.
3. Resume with the original frozen runner from the detached exact-source worktree. In memory only,
   make its existing `_relative` path serializer use the canonical workspace root so rebuilding the
   immutable plan exactly matches attempt 1. Do not patch any scientific function.
4. Allow the original runner to append its single authorized attempt-2 record and complete once.
5. Retain repair scripts, hashes, commands, before/after state, stdout/stderr, and exit status.
6. Run the original independent validator from the frozen source if and only if the runner passes.

## Completion

Repair 3 preserved exactly 615.5649927302729 seconds of consumed active claim time, rebuilt the
immutable plan exactly, and allowed the original runner to complete attempt 2. Runner and independent
validator both exited zero. The final Step-4 classification is `inconclusive_does_not_advance`; see
[`docs/M7_COUNTERFACTUAL_ROLLOUT_RESULTS.md`](../../docs/M7_COUNTERFACTUAL_ROLLOUT_RESULTS.md).

This was the third and final trivial repair allowed for Step 4. No further repair was required.
