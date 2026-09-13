# Approved trivial repair 2: Step-4 target publication accounting

**Program:** `m7-stronger-policy-program-v1`
**Step:** 4 of 5
**Cycle:** `m7-counterfactual-rollout-supervision-v1`
**Repair:** `trivial-repair-2`
**Status:** Approved; execution pending
**Approved:** September 13, 2026
**Original claim source:** `576b896b305273eb29701521d38791712b160a95`
**Repair policy:** [`M7_TRIVIAL_REPAIR_POLICY_V2.md`](M7_TRIVIAL_REPAIR_POLICY_V2.md)

## Frozen scientific plan

Do not alter plan `d9522b85e3112756710a8d3e5ddd3729f35fe7ba998b586fb10c9a17878b0625`,
input manifest `53ba97aed7c40a87b25f5adc1b7925e28f4aad49bd0bfb758e78b125b54181e8`,
panels, target seeds, hidden worlds, continuation policies, q0 leaf, depth, target values, loss,
training, arenas, statistics, or gates.

## Repair scope

1. Execute at the original frozen source in an isolated detached worktree linked to the retained run
   and artifact directories.
2. Reuse the existing 280 replicate-1 one-position shards and complete replicate-2/3 one-position
   shards with the original target generator.
3. Require byte-for-byte replicate-1 equality with the original 14 stratum shards.
4. Publish the staged target only when integer counts satisfy
   `terminal_count + leaf_count == sample_count`, fractions are finite, and each serialized fraction
   matches its canonical derivation within one IEEE-754 ULP. Do not modify any target row or stored
   sample.
5. Retain repair evidence, then resume the original claim runner once and run the original
   independent validator.

The one-ULP acceptance applies only to derived summary metadata. Target float32 values, feature bytes,
sample JSON, action JSON, row ordering, digests, and all scientific outputs remain exact.

## Completion

If runner and independent validation pass, Step 4 receives its normal scientific classification and
Step 5 may proceed. If another trivial packaging/validation issue occurs, the program may use at
most one additional trivial repair under the new three-repair limit. Any scientific change remains
prohibited.
