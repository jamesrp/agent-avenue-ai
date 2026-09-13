# Project status

**As of:** September 13, 2026
**Current selected hybrid champion:** q0-terminal-safety-v1
**Permanent pure-neural baseline:** historical q0
**Latest completed learned experiment:** [M7 structured model v2](M7_STRUCTURED_MODEL_V2_RESULTS.md)

**Latest completed research cycle:** [Step-4 counterfactual rollout infrastructure failure](M7_COUNTERFACTUAL_ROLLOUT_FAILURE.md)

This document is the concise, living index of completed work and current next steps. Historical
milestone documents describe the intended scope at the time; `README.md` and this file describe the
actual present state.

## Milestone status

| Milestone | Status | Evidence |
| --- | --- | --- |
| [1: deterministic rules engine](MILESTONE1.md) | Complete | Engine, observation boundary, replay fixtures, and rule tests |
| [2: lightweight web QA](MILESTONE2.md) | Complete | Hot-seat UI, private views, seeded games, and HTTP tests |
| [3: baseline agents and arena](MILESTONE3.md) | Complete | [Random/heuristic baseline results](MILESTONE3_RESULTS.md) |
| [4: neural foundation](NEURAL_AI_PLAN.md#milestone-4-safe-encoder-dataset-model-and-checkpoint) | Complete | Encoder, dataset, trainer, checkpoint, and compatibility tests |
| [5: first learned checkpoint](NEURAL_AI_PLAN.md#milestone-5-learned-agent-and-first-usable-checkpoint) | Complete | [q0 result and reproduction contract](MILESTONE5_RESULTS.md) |
| [6: frozen iterative self-play](MILESTONE6.md) | Complete | [q1–q4 decisions, final evaluation, and archive](MILESTONE6_RESULTS.md) |
| [Terminal-safety hybrid](TERMINAL_SAFETY_EXPERIMENT.md) | Complete | [Shielded q0–q4 result, all-pairs diagnostics, and archive](TERMINAL_SAFETY_RESULTS.md) |
| [q0 strength audit](Q0_STRENGTH_AUDIT_RESULTS.md) | Complete | 72,000 fresh games; exact missed-lethal and exploit-surface diagnostics |
| [Terminal-offense confirmation](M7_TERMINAL_OFFENSE_CONFIRM_RESULTS.md) | Complete | 60,000 aligned-RNG games; tactical envelope adopted for steps 2–4 |
| [Population replay v1](M7_POPULATION_REPLAY_RESULTS.md) | Complete; did not advance | Mixed replay beat matched controls but frozen robustness gates failed |
| [Structured model v2](M7_STRUCTURED_MODEL_V2_RESULTS.md) | Complete; did not advance | V2 beat v1 in both data arms; mixed selected for Step-4 development |
| [Counterfactual rollout supervision](M7_COUNTERFACTUAL_ROLLOUT_FAILURE.md) | Trivial repair 2 authorized | Original plan fixed; integer-equivalent target publication pending |
| [7: controlled RL experiments](MILESTONE7.md) | In progress at stronger-policy Step 4 | Step 5 waits for repaired validation |

## Current measured policies

| Policy | Training source | Direct measured result |
| --- | --- | --- |
| `random-v1` | None | Symmetry/control baseline |
| `greedy-public-v1` | Hand-authored public heuristic | 86.25% vs random over 400 games |
| historical q0 | 4,000 epsilon-heuristic games, 62,184 decisions | Locked final: 75.5% vs random and 57.3% vs heuristic over 1,000 games each |
| q0-terminal-safety-v1 | 4,000 shielded epsilon-heuristic games, 65,746 decisions | Fresh strength audit: 81.3% vs random, 68.45% vs heuristic, 54.45% vs historical q0; 19.2%–25.65% vs q1–q4 |

The hybrid q0 paired-bootstrap intervals are 75.3%–80.5% against random, 64.4%–70.4% against the
heuristic, and 54.1%–59.3% against historical q0. Historical and hybrid final blocks use different
fresh seeds, so the improvement over the old aggregate is descriptive; the direct hybrid-versus-
historical-q0 arena is the clean head-to-head comparison.

Validation loss and accuracy measure outcome prediction on held-out games, not playing strength.
Promotion and strength claims come from paired gameplay arenas.

## What q0 is

q0 is an action-conditioned Monte Carlo value model:

```text
Q(current acting-player observation, legal semantic action)
    = probability that the acting player eventually wins
```

It uses the information-safe 87-feature `candidate-public-v1` encoder and the 11,393-parameter
`candidate-mlp-v1` network. It was trained on the action actually selected at each recorded decision,
with the final winner as the target from that decision actor's viewpoint.

Historical q0 remains the permanent pure-neural baseline rather than being overwritten by later
experiments. q0-terminal-safety-v1 uses the same action-conditioned model family and adds the narrow
public immediate-loss veto described in the terminal-safety declaration.

## What Milestone 6 means

Milestone 6 is a project phase, not “generation six.” It applies the same declared policy-improvement
loop to generations q1 through q4:

1. freeze the current incumbent;
2. collect 4,000 epsilon-incumbent self-play games;
3. train one warm-started candidate on that generation;
4. compare candidate and incumbent on fresh paired games;
5. enforce random, heuristic, seat, compatibility, and reproducibility guardrails; and
6. either promote the candidate or explicitly retain the incumbent.

The full q1–q4 chain ran on September 2, 2026. Each proposal beat q0 directly but failed the
predeclared heuristic non-regression guardrail, so every immutable decision retained q0. The chain
ended `budget_exhausted_inconclusive`; it did not meet the practical-equivalence plateau criterion.
A separate q1 regeneration reproduced all semantic identities, and q0 completed a fresh locked final
evaluation. See [the Milestone 6 result](MILESTONE6_RESULTS.md).

The terminal-safety rerun completed on September 3, 2026. The separately trained shielded q0 passed
its gate and beat historical q0 directly. All shielded q1–q4 proposals again dominated their parent
but failed the aligned heuristic non-regression guardrail, so shielded q0 was retained under the
predeclared gate. Across 591,219 audited shielded decisions, the safety invariant held with zero
executed avoidable provable losses; this is retained-record coverage rather than a universal proof.

The additional held-out diagnostic evaluated each `qn` on fresh records from every unordered pair in
`(heuristic, q0, ..., q(n-1))`. It covered 14,000 games and exposed poor prediction transfer to
interactions among rejected candidates. Those records were diagnostic-only and did not change
fitting or promotion. See [the terminal-safety result](TERMINAL_SAFETY_RESULTS.md).

## First Milestone 7 experiment

The first controlled Milestone 7 recipe screen reused the exact retained terminal-safety q1 corpus
and compared three MC-only controls with three paired treatments that first learned all-legal-action
ordinal preferences from `greedy-public-v1`. The treatment did not advance: its direct score against
matched controls was 48.37% with a nested 95% bootstrap interval of [45.67%, 51.10%], and
its heuristic non-regression difference relative to shielded q0 was -18.17 percentage points with a
nested 95% interval of [-24.17, -12.28]. It still beat shielded q0 and random in absolute terms,
illustrating the same specialization pattern rather than repairing it.

All three treatment/control direct point estimates favored the controls. Ranking initializers
achieved about 93.4% validation pair accuracy, but final treatment checkpoints had worse MC
validation loss and failed robustness criteria on this fixed development suite. The experiment does
not show that all ranking supervision is harmful; it rejects this exact parent-initialized pairwise
warm-start recipe on this fixed corpus. The selected champion remains unchanged. See
[`M7_HEURISTIC_RANKING_RESULTS.md`](M7_HEURISTIC_RANKING_RESULTS.md).

## q0 strength and tactical-leak audit

The September 9 audit reused one fresh common 1,000-setup block across all 36 unordered matchups in
a nine-policy field, for 72,000 seat-balanced games and 1,015,338 decisions. Selected q0 remained
strong against the three available anchors but was decisively beaten by every q1–q4 descendant. The
descendants were rejected under prior frozen heuristic non-regression criteria; they must not be
described as absolutely weaker than q0. Fresh q3 had no point-estimate loss against the other seven
core policies, although q1/q4 comparisons were statistically close and all descendants share one
training lineage.

The exact public-information audit found 1,832 guaranteed current-turn wins for q0 and 589 misses
(32.15%). Of those misses, 444 did not win immediately and 119 eventually lost. A diagnostic wrapper
forced these wins, missed zero on its own trajectories, and improved q0 by 0.25–1.60 percentage
points against every shared opponent. It did not materially close the q1–q4 gap. q0 still executed
zero avoidable provable immediate losses, confirming that the new leak is offensive rather than a
failure of the terminal-safety shield.

The audit also found highly predictable recruit responses: q0 took visible Saboteur 0% of the time,
visible Sentinel 95.1%, visible Double Agent 77.6%, and visible Codebreaker 12.4%. The old
Codebreaker-up/Saboteur-down trap persisted descriptively, but true hidden-offer slices are
opponent-known offline diagnostics rather than clean causal estimates. See
[`Q0_STRENGTH_AUDIT_RESULTS.md`](Q0_STRENGTH_AUDIT_RESULTS.md).

## Terminal-offense confirmation

The September 11 confirmation removed the remaining policy-ID RNG caveat from the immediate-win
wrapper comparison. Across 60,000 fresh games, the treatment converted all 3,385 encountered
publicly guaranteed current-turn wins, produced zero false wins or avoidable safety violations, and
had zero exact maximum-logit ties across 455,629 q0-family decisions. The aligned shared-opponent
anchor macro improved by +0.75 percentage points with a 95% interval of +0.60 to +0.908, exceeding
the frozen +0.25-point practical threshold.

All structural checks passed, including independent-production oracle agreement on 851,804
replayed decisions, zero setup overlap across 241 prior corpora, and zero pre-endpoint divergences in
28,000 matched control/treatment games. The terminal-offense envelope is now fixed for stronger-
policy steps 2–4. The selected checkpoint and web default remain q0-terminal-safety-v1 until step 5.
See [`M7_TERMINAL_OFFENSE_CONFIRM_RESULTS.md`](M7_TERMINAL_OFFENSE_CONFIRM_RESULTS.md).

## Population replay v1

Step 2 collected three paired 4,000-game q0-only and mixed-population corpora, trained six unchanged
model-v1 checkpoints from identical q0 tensors, and evaluated them over 24,000 fresh development
games. Mixed replay beat matched controls directly at 62.10% [60.13%, 64.00%], improved matched
heuristic performance by +6.33 percentage points, and improved q1–q4 stress comparisons by roughly
12–14.5 points.

The exact recipe does not advance. Its treatment-minus-parent heuristic interval was -5.056 to
+1.278 points against a strict lower-bound requirement above -5. The implemented seat rule also
failed because it included the losing control side of the direct arena; treatment's own lowest seat
point was 54.0%. One validator-only repair corrected nested-bootstrap RNG consumption order without
changing claim artifacts or the decision. Both data arms remain fixed for Step 3. See
[`M7_POPULATION_REPLAY_RESULTS.md`](M7_POPULATION_REPLAY_RESULTS.md).

## Structured model v2

Step 3 re-encoded both Step-2 data arms with a 519-feature public-history/consequence representation
and trained six 35,779-parameter q0-embedded residual models. V2 beat matched v1 controls on q0-only
data at 53.93% [51.47%, 56.63%] and mixed data at 55.80% [52.73%, 58.67%]. Mixed-v2 beat q0-only-v2
at 64.63% [59.07%, 69.07%]. The data-by-architecture interaction was inconclusive.

Both data arms passed architecture, q0-parent, random, candidate-seat, tactical, and integrity gates,
but failed the q0-parent heuristic non-regression interval. The exact recipe does not advance. The
frozen robustness-floor rule selects mixed-v2 only as the development input for Step 4. One
operational checksum-log repair left all scientific evidence unchanged; exact-source independent
validation passed. See [`M7_STRUCTURED_MODEL_V2_RESULTS.md`](M7_STRUCTURED_MODEL_V2_RESULTS.md).

## Counterfactual rollout Step-4 failure

Step 4 implemented a provenance-safe latent state, 14-stratum public panel, 10-world depth-nine q0-
leaf teacher, paired rollout auxiliary trainer, and independent numeric target validator. Its smoke
passed at roughly 160 transition/leaf units per second and projected inside the compute budget.

The claim did not reach training. Attempt 1 failed because claim shards contained 20 positions per
stratum while the assembler required one position per shard. The sole repair regenerated all 280
replicate-1 positions and verified all 1,400 rows exactly, but refused publication when direct
terminal-count division and `1 - leaf_fraction` differed by one floating-point ULP. No canonical
three-replicate targets, checkpoints, arenas, statistics, selection, or scientific result yet exists.
On September 13, the user approved up to three strictly mechanical repairs per step and authorized
repair 2 to validate integer-equivalent one-ULP metadata without changing target rows or the frozen
plan. Step 5 remains waiting. See
[`M7_COUNTERFACTUAL_ROLLOUT_FAILURE.md`](M7_COUNTERFACTUAL_ROLLOUT_FAILURE.md).

## Web status

The QA web UI currently supports:

- human versus human;
- human versus random;
- human versus `greedy-public-v1`;
- human versus historical q0;
- human versus `q0-terminal-safety-v1`; and
- either human seat for every automated mode.

Learned policies are configured by opaque server-side allowlist keys. The server validates and
caches immutable inference checkpoints; the browser receives the public label and checkpoint
fingerprint, never a filesystem path, tensor digest, logits, or private game/model state. See
[`WEB_QA.md`](WEB_QA.md).

## Reproducibility status

The repository currently records:

- normalized game, agent, encoder, model, training, and iteration configurations;
- domain-separated setup, agent, split, training, arena, and bootstrap seeds;
- rules, code, corpus, dataset, model tensor, checkpoint, and report fingerprints;
- semantic actions in retained corpora and game records sufficient to replay those completed games;
- deterministic game-level data splits;
- paired, seat-swapped arena blocks;
- immutable promotion decisions;
- replay-derived immediate-loss safety and guaranteed-current-turn-win diagnostics;
- fresh common-block all-pairs policy tournaments; and
- held-out, setup-disjoint all-pairs prior-policy evaluation plans and reports.

The q0 source corpus, dataset, checkpoint, logs, and reports remain retained. The q0 historical arena
aggregates predate the individual-record policy. Every Milestone 6, terminal-safety, M7 ranking, and
q0 strength-audit arena retains compressed semantic records. The strength audit adds 72,000 fresh
games, exact information-safe lethal conversion, matched terminal-offense contrasts, repeatable
recruit-response slices, and full independent recomputation. These completed lines are packaged in
verified ignored archives with embedded member checksums. Generated artifacts stay out of ordinary
Git history; compact results and exact source revisions remain committed. See the
[q0 strength audit](Q0_STRENGTH_AUDIT_RESULTS.md),
[M7 ranking result](M7_HEURISTIC_RANKING_RESULTS.md),
[terminal-safety result](TERMINAL_SAFETY_RESULTS.md), [Milestone 6 result](MILESTONE6_RESULTS.md),
and [experiment protocol](EXPERIMENT_PROTOCOL.md).

## Verified source revisions

| Purpose | Revision | Package code fingerprint |
| --- | --- | --- |
| q0 corpus, dataset, and training | `7fe2e5af6339efc86db285349043bb16b298eed1` | `b92b07727d89eb2f9471316fa44992a2eb337da2186a25f3bc4637404bb09d8d` |
| q0 learned-agent arena evaluation | `d46350356a7bf461468ba33c6bf1f28b3231cd14` | `6909e9316510f48cad515b99a8f9b19288ee5c43d5925f84b88ad853479031bc` |
| Milestone 6 production q1–q4 and final evaluation | `139318bad909438e8a3e1cb9dd962c80653875d0` | `5a66a4a63f7c5680f3db81e83dfebf53af63f73d29b653645850bb013a45ba0c` |
| Terminal-safety q0–q4, all-pairs diagnostics, and final | `19c2871080503c62a53522415e0645913d7674b0` | `9c8b39bb4e81bb29e0902c56198557f63b7e9992bdaff845967c562fc064f429` |
| M7 heuristic ranking warm-start repaired run | `240d33495c29e227c57afe2b58d73ad792fbdbd8` | `9f9e78d1c4a5a124a12d8f01056788520a65bc86bec652ffa113da24e5975b49` |
| q0 strength and tactical-leak fresh run | `2cefd151383e9cf3a614e2b58ee3ac2450727a61` | `ae8cd87f4b60c4b32a2b0ac728b1c3c0ec51f74944b9afb4d5089a1547d2e4e5` |
| Terminal-offense aligned-RNG confirmation | `5bbda9e18d23920f5ad7686e055615aad703c097` | `5dd1d77e67804cf959ccd9a90dd9569a54924623439a30194111da3304c3b4f6` |
| Population replay v1 claim run | `fde19b5d3c327e539c29913a973b5e4d76ffff5b` | `e0adb698b5329031ad077f155ae5911f982b1d235bde21474f7262fe26694459` |
| Structured model v2 claim run | `a63f0819dd679308fe03ec23d619e609529b1166` | `82b5111694299b8f24b848222449946508592478fca50d2b2bca4a60139f8911` |

## Research workflow status

A bounded coordinator and cycle-record convention are implemented and tested; see
[`RESEARCH_WORKFLOW.md`](RESEARCH_WORKFLOW.md) and
[`RESEARCH_WORKFLOW_INVENTORY.md`](RESEARCH_WORKFLOW_INVENTORY.md). It records queued/running/
completed/failed/blocked/stopped work, exact attempts and exit status, output evidence, bounded
retries, source freezing, stop/budget state, and one optional Shelley completion message. It wraps
rather than duplicates the existing experiment runners.

The setup-only declarations are `research/cycles/setup-smoke-v1.json` through v3. See
[`RESEARCH_WORKFLOW_SETUP_REPORT.md`](RESEARCH_WORKFLOW_SETUP_REPORT.md).

The first approved retrospective cycle, `m7-diagnostic-readiness-v1`, is complete with one declared
operational repair. The repaired run verified 3 archives, all 93 live retained corpora, 90 restored
corpora, and 38 arena aggregates with zero mismatch; it also implemented and exercised allowlisted
historical/current learned web opponents. It recommends counterfactual candidate-ranking
supervision as the most directly isolating next **question**, with mixed-opponent replay a close
second. This is a diagnostic recommendation, not an authorization or strength result. See
[`M7_DIAGNOSTIC_READINESS_RESULTS.md`](M7_DIAGNOSTIC_READINESS_RESULTS.md).

The approved ranking cycle completed through one declared operational repair. The original run
stopped on a runtime-only `Tensor` import defect after producing valid replicate-1 artifacts. The
repair changed only initializer loading and added a regression test; original and repaired
replicate-1 tensors match exactly. The repaired run retained 22 arenas containing 15,600 games,
recomputed every aggregate and safety report, and concluded that the frozen recipe does not advance.
See [`M7_HEURISTIC_RANKING_RESULTS.md`](M7_HEURISTIC_RANKING_RESULTS.md).

The approved q0 strength audit completed through one validator-only operational repair. Its 72,000
fresh games and exact tactical audits reproduced independently. The repair corrected only the
validator's result-fingerprint normalization and did not change claim-generating evidence. See
[`Q0_STRENGTH_AUDIT_RESULTS.md`](Q0_STRENGTH_AUDIT_RESULTS.md).

The approved terminal-offense confirmation completed without a repair. Its 60,000 fresh games and
independent validation passed every structural criterion and the separate practical-lift threshold.
The hard envelope is fixed for steps 2–4 of the approved stronger-policy program. See
[`M7_TERMINAL_OFFENSE_CONFIRM_RESULTS.md`](M7_TERMINAL_OFFENSE_CONFIRM_RESULTS.md) and the living
[`stronger-policy progress log`](../research/cycles/M7_STRONGER_POLICY_PROGRESS.md).

The approved population-replay cycle completed after one validator-only repair. The claim runner
retained 24,000 training games, six datasets/checkpoints, and 24,000 development games. Repaired
validation reproduced every artifact and the unchanged `does_not_advance` decision. See
[`M7_POPULATION_REPLAY_RESULTS.md`](M7_POPULATION_REPLAY_RESULTS.md).

The approved structured-model cycle completed after one operational checksum-log repair. Its six
v2 fits, 30,000 fresh games, local 519-feature reconstruction, complete local arena aggregation, and
selection reproduced. Mixed-v2 is the fixed development input for Step 4, not a promoted policy. See
[`M7_STRUCTURED_MODEL_V2_RESULTS.md`](M7_STRUCTURED_MODEL_V2_RESULTS.md).

The stronger-policy program has reopened Step 4 under the September 13 trivial-repair policy. Repair
2 preserves the original source/plan/panels/targets and changes only target publication accounting.
Step 5 remains waiting for a completed validated Step-4 result. See
[`M7_TRIVIAL_REPAIR_POLICY_V2.md`](../research/cycles/M7_TRIVIAL_REPAIR_POLICY_V2.md),
[`M7_COUNTERFACTUAL_ROLLOUT_REPAIR2.md`](../research/cycles/M7_COUNTERFACTUAL_ROLLOUT_REPAIR2.md),
and the living [`stronger-policy progress log`](../research/cycles/M7_STRONGER_POLICY_PROGRESS.md).

Step-4 repair 2 is the current authorized research operation. VM-reboot execution resume and off-VM
artifact durability remain unconfigured.

## Planned next work

1. execute Step-4 trivial repair 2 at the original frozen source, preserving every target row and
   scientific setting;
2. resume the original Step-4 runner and independent validator only after exact repair evidence;
3. freeze and run Step 5 if repaired Step 4 produces a completed selection artifact; and
4. choose an off-VM retention mechanism if stronger disaster recovery is desired.
