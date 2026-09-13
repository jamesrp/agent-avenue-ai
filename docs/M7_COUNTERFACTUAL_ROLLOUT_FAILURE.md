# Counterfactual rollout supervision: infrastructure failure

**Program:** `m7-stronger-policy-program-v1`
**Step:** 4 of 5
**Cycle:** `m7-counterfactual-rollout-supervision-v1`
**Status:** Historical initial block; trivial repair 2 authorized September 13, 2026
**Initial stop:** September 12, 2026
**Reauthorization:** [`research/cycles/M7_COUNTERFACTUAL_ROLLOUT_REPAIR2.md`](../research/cycles/M7_COUNTERFACTUAL_ROLLOUT_REPAIR2.md)
**Agreement:** [`research/cycles/M7_COUNTERFACTUAL_ROLLOUT_SUPERVISION_V1.md`](../research/cycles/M7_COUNTERFACTUAL_ROLLOUT_SUPERVISION_V1.md)
**Claim source:** `576b896b305273eb29701521d38791712b160a95`

## Decision

Stop the stronger-policy program before Step 5. Step 4 produced no validated rollout-supervised
checkpoint, arena, statistic, selection, or strength result. The single authorized operational
repair was exhausted and no second repair is permitted under the approved agreement.

Do not classify the rollout recipe as stronger, weaker, failed, or inconclusive. The correct status
is infrastructure-blocked before scientific evaluation.

## What completed

The preclaim implementation and smoke were healthy:

- 315 tests passed at freeze;
- production and independent target throughput were about 160 transition/leaf units per second;
- projected full execution was 204.73 minutes, below the 465-minute cutoff;
- smoke leaf fractions were below 50% with zero mechanical cap errors;
- exact Step-3 input/control reproduction and independent numeric target validation passed; and
- claim setup holdout, panel feasibility, and source/input identities passed.

The claim then retained:

- immutable source, plan, input audit, and source identity;
- all three complete public-safe panels, each 280 positions across 14 strata; and
- 14 replicate-1 multi-position target shards containing 1,400 candidate rows.

No training or evaluation was reached.

## Attempt 1 failure

Claim mode correctly selected 20 positions per stratum. The runner saved one target shard per
stratum, but the frozen assembler required each resumable shard to contain exactly one panel
position. It stopped with:

```text
RolloutArtifactError: a resumable target shard must contain exactly one panel position
```

No canonical target artifact was published.

## Sole operational repair

A standalone exact-source repair regenerated one-position packaging without changing the frozen
source, plan, panel, worlds, policies, seeds, target generator, or target rows.

For replicate 1 it:

- generated all 280 one-position shard pairs;
- compared every regenerated row with the original stratum shards; and
- verified exact equality for all 1,400 feature bytes, float32 target bytes, sample JSON bytes, and
  action JSON bytes.

It assembled a non-published staging target with:

| Item | Count |
| --- | ---: |
| Positions | 280 |
| Candidate rows | 1,400 |
| World samples | 14,000 |
| Depth leaves | 5,853 |
| Terminal samples | 8,147 |
| Mechanical cap errors | 0 |

The repair then refused publication because its metadata check required exact floating equality:

```text
serialized terminal fraction = 1 - 5853/14000
                             = 0.5819285714285714
counted terminal fraction    = 8147/14000
                             = 0.5819285714285715
```

The difference is one floating-point ULP (`1.11e-16`), not a target-row or accounting disagreement.
The strict repair guard nevertheless stopped as designed. Replicates 2 and 3 were not generated, and
normal runner resume never began.

## Scientific boundary

Permissible conclusions are limited to infrastructure:

- all three public panel selections satisfied their frozen quotas;
- replicate-1 target rows were deterministic across both shard organizations;
- the staged target's leaf fraction was 41.81%, under the 50% gate; and
- artifact publication failed for packaging/float-validation reasons.

Not permissible:

- target-quality claims;
- control or treatment training claims;
- rollout-treatment playing strength;
- Step-4 advancement or non-advancement;
- Step-5 promotion disposition; or
- using partial target shards as model training inputs.

## Step-5 disposition

Step 5 is blocked and was not executed. The agreement's fallback Step-5 descriptive league assumed a
completed Step-4 gate/selection artifact; none exists. No rollout-supervised family can enter a final
league, and proceeding would violate the approved sequential program.

A future separately approved repair could compare integer leaf/terminal/sample counts, or explicitly
accept this bounded one-ULP complementary-fraction difference, while reusing the unchanged plan and
target rows. On September 13, the user granted that approval under the new trivial-repair policy.
This document remains the immutable explanation of the original stop; subsequent completion, if any,
will be recorded separately.

## Retained partial evidence

| Item | Identity |
| --- | --- |
| Claim source | `576b896b305273eb29701521d38791712b160a95` |
| Package code fingerprint | `81913c1e52a22776fde97661936b7793f51f6c6882ce418aa76220a03de258d7` |
| Plan fingerprint | `d9522b85e3112756710a8d3e5ddd3729f35fe7ba998b586fb10c9a17878b0625` |
| Input manifest | `53ba97aed7c40a87b25f5adc1b7925e28f4aad49bd0bfb758e78b125b54181e8` |
| Input audit | `5591b0a4da966b75709cc0b63ac3b10ec09cc69551c1933be5dfd08c416e5abd` |
| Staged NPZ | `243245b7e494e57b1f023aa92007764c81adcc6c2ce2120b91d2eea2d745ac24` |
| Staged manifest | `6506b42417ecbe22942622ecac316139de81f8f00c00b683f4c3373614d193ba` |
| Repair script | `46bab7c27adda06b8a2955760b87f51e84f1be85a1bf4a36b81611d6bbbde5d0` |
| Partial archive manifest | `2b8c6932ccf425d9a31f594cd6a5b4a39002b58678a0657fb75c90c70bb7ddb7` |
| Partial archive SHA-256 | `1f89a46205a3622712a575558969eea0ad89f93a86ee212e6ced3fb6e24b46c8` |

Partial evidence is under `runs/m7-counterfactual-rollout-supervision-v1/`. The verified archive is
`artifacts/archive/m7-counterfactual-rollout-supervision-v1-incomplete-2026-09-12.tar.gz` with 614
checksummed payload files. All staged/repair targets are explicitly noncanonical and must not be used
for training or live inference.
