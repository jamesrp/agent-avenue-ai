# Structured public-history model v2: results

**Program:** `m7-stronger-policy-program-v1`
**Step:** 3 of 5
**Cycle:** `m7-structured-model-v2`
**Status:** Complete; structured recipe does not advance
**Completed:** September 12, 2026
**Agreement:** [`research/cycles/M7_STRUCTURED_MODEL_V2.md`](../research/cycles/M7_STRUCTURED_MODEL_V2.md)
**Operational repair:** [`research/cycles/M7_STRUCTURED_MODEL_V2_REPAIR1.md`](../research/cycles/M7_STRUCTURED_MODEL_V2_REPAIR1.md)
**Claim source:** `a63f0819dd679308fe03ec23d619e609529b1166`

## Decision

The combined structured-v2 recipe beat matched model-v1 controls in both retained data arms, but
neither arm passed the frozen q0-parent heuristic non-regression gate. The Step-3 classification is
`does_not_advance`; no checkpoint or web default changes.

The frozen Step-4 selection rule chooses **mixed-population structured v2 (M)** as a
**development-selected input**, not an advancing or promoted policy. Its robustness floor was -3.833
percentage points versus -7.0 for q0-only structured v2.

## Design

Step 3 was a predeclared 2 × 2 data-by-architecture study:

| Data | Model v1 | Structured v2 |
| --- | --- | --- |
| q0-only | three retained Step-2 controls | three new v2 fits |
| mixed population | three retained Step-2 controls | three new v2 fits |

The six v2 models used a 519-feature safe encoder: the unchanged 87-feature prefix, eight completed
public turns, exact public play-consequence branches, and recruit support envelopes over every
public-consistent hidden identity. The 35,779-parameter residual model embedded q0 exactly at
initialization and kept its base trainable. Selected-action terminal Monte Carlo BCE, splits,
training shuffle seeds, optimizer, and tactical envelope were unchanged.

The run re-encoded all six Step-2 corpora, trained six v2 checkpoints, and retained **30,000 fresh
development games** across 60 arena cells.

## Architecture effects

| Estimand | Point estimate | Nested 95% interval |
| --- | ---: | ---: |
| q0-only v2 versus matched v1 | **53.93%** | 51.47% to 56.63% |
| mixed v2 versus matched v1 | **55.80%** | 52.73% to 58.67% |
| Equal-arm pooled architecture effect | **54.87%** | 52.50% to 56.85% |
| Mixed-v2 versus q0-only-v2 | **64.63%** | 59.07% to 69.07% |
| Data × architecture interaction | +1.87 pp | -1.50 to +5.57 |

All six direct architecture replicate points favored v2:

- q0-only: 52.2%, 53.0%, and 56.6%;
- mixed: 52.7%, 58.5%, and 56.2%.

This supports the **combined encoder-plus-residual-model recipe**. It does not isolate public history,
candidate consequences, phase routing, or parameter count: v2 has 35,779 parameters versus v1's
11,393.

The interaction interval crosses zero, so the result does not establish that mixed data benefits
more from structured v2 than q0-only data does.

## External robustness

| Statistic | q0-only v2 | Mixed v2 |
| --- | ---: | ---: |
| Versus selected q0 parent | **87.33%** [85.57%, 88.97%] | **82.53%** [80.30%, 85.07%] |
| Versus random | **86.92%** [84.58%, 89.25%] | **89.42%** [86.83%, 92.00%] |
| Heuristic minus q0-parent reference | **-8.00 pp** [-12.00, -4.22] | **-3.56 pp** [-8.83, +1.67] |
| Minimum v2 candidate seat | 48.4% | 51.6% |
| Equal eight-opponent macro | 70.07% | 76.94% |

Both arms passed direct architecture, q0-parent, random, seat, tactical, and integrity conditions.
Both failed only the heuristic non-regression rule, which required the lower endpoint to exceed -5
points. Mixed was closer but still failed because its lower endpoint was -8.83.

The eight-opponent macro is descriptive and selection-only. Four components are q1–q4, correlated
descendants and members of the mixed replay lineage; the macro is not independent generalization
evidence.

## Training results

| Replicate | C best epoch / validation loss | M best epoch / validation loss |
| --- | ---: | ---: |
| 1 | 29 / 0.51196 | 18 / 0.50157 |
| 2 | 18 / 0.52877 | 25 / 0.49804 |
| 3 | 30 / 0.51390 | 23 / 0.48925 |

Every v2 checkpoint retained q0 parent lineage, the corresponding frozen Step-2 data/split identity,
arm-independent structured initialization per replicate, paired C/M shuffle seeds, and fresh AdamW
state. Within-distribution loss is diagnostic only; gameplay provides the strength evidence.

## Information safety and validation

The production encoder accepts only a public observation and legal semantic candidate. It imports no
`GameState` or transitions. Play terminal features use pure public-material adjudication; recruit
features enumerate public-consistent hidden support without accessing the true face-down card.

The independent validator:

- reconstructed all 519 features without importing the production v1/v2 encoders;
- reproduced six dataset row orders, labels, and splits;
- checked q0 initialization and checkpoint lineage;
- locally recomputed all 60 complete arena reports from records, including schedules, seats, policy
  configs, RNG identities, margins, terminal reasons, turns, and decisions;
- replayed tactical invariants; and
- reproduced nested statistics and the Step-4 selection.

All tactical checks passed: no enveloped v1/v2 policy executed an avoidable immediate loss or missed
a publicly guaranteed current-turn win.

## Operational repair

The claim runner completed normally. Initial validation stopped because wrapper-owned
`driver.stdout` changed after the runner had checksummed it as empty. The one operational repair
restored that mutable log to the already-declared empty SHA-256 state and reran validation at the
exact claim source. No scientific artifact or source changed; validation passed.

After validation, wrapper-owned stdout/stderr/exit files were excluded from future immutable checksum
scopes and a regression test was added. That prevention commit is not part of the claim source.

## Runtime

The claim runner completed in about 2 hours 22 minutes. Exact-source independent validation completed
in about 1 hour 14 minutes. Combined execution was about **3 hours 36 minutes**, inside the eight-hour
step budget.

## Step-4 implication

Structured v2 improves action ranking consistently, but the heuristic confidence guard remains the
unresolved weakness. The selected mixed-v2 recipe is therefore the right development base for
counterfactual all-legal-action rollout supervision: it preserves mixed replay's direct and
descendant strength while providing the representation needed to learn from alternative-action
values.

## Provenance

| Item | Identity |
| --- | --- |
| Claim source | `a63f0819dd679308fe03ec23d619e609529b1166` |
| Encoder schema fingerprint | `66d806ce7ba667a716cf31be2210e449feabcbf06e90bb6a13754a0909e4c628` |
| Plan fingerprint | `edc24d7c1f6a67bf2860bdc3dc401712152b05d7f08c17782045ea284594660a` |
| Result fingerprint | `346bb85a4ea6c31f6f9ef79671936fdc0c11043f8b1b011380a2589c95e14fac` |
| Statistics | `eadf71e10ede64cdd3ebc9d976b78be31518460cac8abaef0521b6677487a9c2` |
| Safety | `be3fee0023eed188c1c989ae663726f6a9f337eb437d6ed82b5a219585d38da5` |
| Selection | `5d33deb44583ca58114fc2c30ef50e40e0d215be66d054ae1a8f5d9495fedc08` |
| Independent validation | `679e297b14d5dfd5c122860eeebb629a305a8ca8191450323aca99d6d87eafc9` |
| Archive manifest | `0686c22119f6978efc606885d1e6dac5576690f73800cc310ad0e7087ae2c5e8` |
| Archive SHA-256 | `ad42072825d110b10ea878d00ea0232b66b17f817d0f4f1f5e7fb6eb46aab669` |

Retained evidence is under `runs/m7-structured-model-v2/`. The verified archive is
`artifacts/archive/m7-structured-model-v2-2026-09-12.tar.gz` with 243 checksummed payload files.
