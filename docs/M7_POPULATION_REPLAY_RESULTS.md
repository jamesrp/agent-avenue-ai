# Population replay v1: results

**Program:** `m7-stronger-policy-program-v1`
**Step:** 2 of 5
**Cycle:** `m7-population-replay-v1`
**Status:** Complete; mixed-population model-v1 collection does not advance
**Completed:** September 12, 2026
**Agreement:** [`research/cycles/M7_POPULATION_REPLAY_V1.md`](../research/cycles/M7_POPULATION_REPLAY_V1.md)
**Validator repair:** [`research/cycles/M7_POPULATION_REPLAY_V1_REPAIR1.md`](../research/cycles/M7_POPULATION_REPLAY_V1_REPAIR1.md)
**Claim source:** `fde19b5d3c327e539c29913a973b5e4d76ffff5b`

## Decision

Under the frozen conjunctive rule, mixed-population model-v1 collection **does not advance**. It
produced a large and consistent direct improvement over size-matched q0-only retraining and was much
stronger against q1–q4, but it narrowly failed the q0-parent heuristic non-regression confidence
bound. The implemented all-candidate seat condition also failed, although that specific failure came
from the losing control side of the direct treatment-control arena rather than a treatment seat
collapse.

Both q0-only and mixed-population corpora, datasets, and all six v1 checkpoints remain fixed inputs
for the Step-3 data-by-architecture experiment. No champion or web default changed.

## Design and retained evidence

Three paired corpus/training replicates each compared:

- 4,000 q0-only control games; and
- 4,000 mixed-population treatment games with exact 40% q0 and 10% each
  q1/q2/q3/q4/heuristic/random logical-slot marginals.

Control and treatment shared setup blocks, physical seats, logical lane RNG streams, q0 tensor
initialization, split seed, and training shuffle seed within each replicate. The encoder, 11,393-
parameter MLP, selected-action terminal Monte Carlo target, BCE loss, optimizer, training budget,
epsilon 1/5, and offense+safety envelope were unchanged.

The cycle retained **24,000 training games**, six datasets, six checkpoints, and **24,000 fresh
development games** across 54 arenas. The setup holdout covered 12,300 proposed identities and 60,104
prior identities from 271 retained corpora, with zero overlap.

## Primary result

| Replicate | Mixed treatment vs q0-only control |
| --- | ---: |
| 1 | 60.5% |
| 2 | 63.3% |
| 3 | 62.5% |
| **Nested result** | **62.10% [60.13%, 64.00%]** |

All three independent corpus/training replicates favored population replay. This is strong evidence
that opponent diversity materially changes and improves the resulting model against its matched
q0-only controls under this policy field.

## External and stress comparisons

| Statistic | Point estimate | Nested 95% interval |
| --- | ---: | ---: |
| Treatment vs selected q0 parent | **77.20%** | 74.40% to 79.63% |
| Control vs selected q0 parent | **80.20%** | 78.20% to 82.07% |
| Treatment vs random | **86.25%** | 83.58% to 88.67% |
| Treatment minus control vs heuristic | **+6.33 pp** | +2.83 to +9.94 |
| Treatment minus parent vs heuristic | **-1.89 pp** | **-5.056 to +1.278** |
| Treatment minus control vs q1 | **+12.50 pp** | +8.33 to +17.00 |
| Treatment minus control vs q2 | **+14.50 pp** | +7.00 to +23.33 |
| Treatment minus control vs q3 | **+12.50 pp** | +7.00 to +17.83 |
| Treatment minus control vs q4 | **+12.17 pp** | +7.00 to +17.00 |
| Treatment minus control vs historical q0 | +0.25 pp | -2.17 to +2.58 |
| Treatment minus control vs random | -0.17 pp | -3.33 to +3.33 |

The treatment-parent heuristic criterion required a lower endpoint strictly above -5 percentage
points. Its lower endpoint was -5.056, missing by **0.056 points**. The interval includes both modest
regression and improvement; the result is a failed non-inferiority demonstration, not positive proof
that mixed replay harms heuristic play.

The descriptive equal-opponent macro was 72.22% for treatment and 65.33% for control. It should not
be treated as a rating: half its external-policy weight comes from q1–q4, which are correlated
q0-family descendants and treatment-population members.

## Seat condition qualification

The frozen minimum across all candidate seat rows was 33.8%, below the required 45%. That value was
not a treatment seat collapse. It was the control candidate's derived Player-One rate in the
replicate-3 direct treatment-control arena, where treatment won 66.2% from the opposite seat.
Treatment's lowest observed seat point estimate was 54.0%.

The formal gate still fails because it was frozen and applied to both sides of the direct arena.
Future protocols should apply candidate seat floors to external-anchor cells rather than making a
large direct win mechanically create a failure through the losing control arm.

## Training behavior

Mixed-population checkpoints had lower within-distribution validation loss and higher validation
accuracy in every replicate:

| Replicate | Control best epoch / loss | Treatment best epoch / loss |
| --- | ---: | ---: |
| 1 | 29 / 0.51561 | 41 / 0.50003 |
| 2 | 18 / 0.53396 | 25 / 0.50942 |
| 3 | 35 / 0.52307 | 32 / 0.50039 |

These losses are not direct cross-arm quality estimates because control and treatment validation
sets come from different behavior distributions. Gameplay supplies the strength evidence.

All candidates began from q0 tensor digest
`b1562237220e3a59b1e514327f3a5acc3d60767676befc69970f88ef98253767` with paired control/treatment
shuffle seeds and fresh AdamW state. Natural paired-block validation sizes were 202, 198, and 198 in
replicates 1–3; control and treatment pair IDs matched exactly within each replicate.

## Tactical and integrity results

Every training and arena record passed the fixed envelope invariants: zero avoidable immediate
losses and zero missed publicly guaranteed wins for enveloped policies. Corpus assignments, exact
per-seat marginals, split groups, initialization, training seeds, checkpoint lineage, 54 arena
aggregates, shared opponent blocks, and the three-way heuristic reference blocks validated.

## Validator-only repair

The claim runner completed and emitted its immutable result at source `fde19b5`. The first
independent validation failed because its nested bootstrap interleaved outer replicate draws with
inner block draws, while production first materialized all three outer draws. Point estimates,
games, models, reports, thresholds, and the `does_not_advance` decision were unchanged.

The one authorized repair changed only the validator script, its regression tests, and the committed
repair agreement. Repaired validation at source `2bf8822` required the original package code, rules,
and lock fingerprints, verified the exact changed-path allowlist, reproduced all retained evidence,
and passed. No claim game, model, arena, statistic, or result was regenerated.

## Interpretation for Step 3

Population replay clearly teaches a different and often stronger response distribution: it beats
matched q0-only models directly and closes much of their q1–q4 weakness while improving heuristic
performance relative to those controls. It does not by itself remove all robustness tradeoffs, and
the current action-conditioned v1 representation may still be the limiting factor.

Step 3 therefore keeps both data arms and asks whether public history, explicit safe candidate
consequences, and separate play/recruit computation preserve the mixed-data gains while improving
parent/heuristic robustness. This avoids post-result corpus selection and allows a genuine
data-by-architecture interaction analysis.

## Provenance

| Item | Identity |
| --- | --- |
| Claim source | `fde19b5d3c327e539c29913a973b5e4d76ffff5b` |
| Repair validation source | `2bf8822e3247a104870e55746941d438e3f619b0` |
| Package code fingerprint | `e0adb698b5329031ad077f155ae5911f982b1d235bde21474f7262fe26694459` |
| Plan fingerprint | `a85c4d352386be2512a71dd08326abe45a3f80cca5a762556c749e3f02408a31` |
| Result fingerprint | `aaf30bf3b6939f964a8c69a29a60e911859c253ca5e20517091f4b7460c954f2` |
| Repaired validation | `1930ffc5df1378bbc0daad484d1140707553b3f4b69fe7d4c869abf2589905e2` |
| Input audit | `090bba49c62e711a87f9f932ed504c04ea086289ee8e529cf0e8af752b7bd919` |
| Setup holdout | `ef07aed789d07a14022d008b2b46953383d8b6307038fb323d57ee50177cc577` |
| Split alignment | `9653dfdf1e7cb7f401af753ccfe8e90ce270a7fc12d97c98f097860c4c774552` |
| Training summary | `b8f5f2c74734ff2c1571ccc025394c7f2fb8bdf9b17d44329fbd65d29c941a1c` |
| Statistics | `78901cd542e8baed6de12688636ac74324777654e36a6e447a8957c82bce3510` |
| Archive manifest | `eab75b8d1d448a710d1b70f9355928262b4cabb242e319a2b00c6bbadfa6a9e9` |
| Archive SHA-256 | `993436e375e02816fb0a83e3009636c1538b0c0000e874dc0913cb20cc1632eb` |

Retained evidence is under `runs/m7-population-replay-v1/`. The verified archive is
`artifacts/archive/m7-population-replay-v1-2026-09-12.tar.gz` with 232 checksummed payload files.
