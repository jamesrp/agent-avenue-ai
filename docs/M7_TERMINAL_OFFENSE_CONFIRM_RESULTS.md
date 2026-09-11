# Terminal-offense confirmation: results

**Program:** `m7-stronger-policy-program-v1`
**Step:** 1 of 5
**Cycle:** `m7-terminal-offense-confirm-v1`
**Status:** Complete; structurally adopted for steps 2–4
**Completed:** September 11, 2026
**Agreement:** [`research/cycles/M7_TERMINAL_OFFENSE_CONFIRM_V1.md`](../research/cycles/M7_TERMINAL_OFFENSE_CONFIRM_V1.md)
**Claim source:** `5bbda9e18d23920f5ad7686e055615aad703c097`

## Decision

Retain `TerminalOffense(TerminalSafety(...))` as a fixed tactical envelope for program steps 2–4.
Every predeclared structural criterion passed, and the separate practical-lift criterion also passed.
This does not promote a checkpoint or change the web opponent; only step 5 may do that.

## Fresh confirmation

The run used two disjoint families of 1,000 paired setup blocks. Thirty cells covered control and
treatment against q1–q4, historical q0, heuristic, and random, plus a descriptive direct cell. The
complete result contains **60,000 games and 851,804 decisions**.

Control and treatment used different policy IDs but one shared q0 RNG identity in matched
shared-opponent cells. Setup, seat, q0 RNG stream, and opponent RNG stream were therefore aligned.
Across 28,000 matched games, 389,540 semantic actions matched before the first guaranteed-win
endpoint, with zero metadata mismatches, pre-endpoint divergences, or treatment endpoint failures.

## Structural evidence

| Check | Result |
| --- | ---: |
| Independent-production oracle agreement | 851,804 / 851,804 decisions |
| Candidate actions classified | 4,097,871 |
| Exact engine play-transition cross-checks | 909,772 |
| Treatment guaranteed wins converted | **3,385 / 3,385** |
| Treatment false guaranteed wins | **0** |
| Control guaranteed wins missed | 1,134 / 3,597 |
| Avoidable immediate-loss violations | **0** in both arms |
| Exact maximum-logit ties | **0** in 455,629 q0-family decisions |
| Selected actions outside exact maxima | **0** |
| Prior setup overlap | **0** across 241 corpora / 258,024 records |

At guaranteed-win endpoints in matched games, control and treatment both converted 1,970 times.
Control missed while treatment converted 900 times. The remaining matched games reached no such
endpoint before their histories ended or diverged after an earlier intervention.

The zero exact-tie result removes the outstanding policy-ID tie-breaking concern for this checkpoint
and retained distribution. It is not a universal claim about future models.

## Practical effect

The primary anchor macro equally weighted historical q0, heuristic, and random. Its treatment-minus-
control effect was **+0.75 percentage points**, with a stratified common-block 95% interval of
**+0.60 to +0.908 points**. The frozen practical criterion required the lower endpoint above +0.25
points and passed.

| Shared opponent | Treatment minus control | 95% interval |
| --- | ---: | ---: |
| heuristic | **+1.25 pp** | +0.925 to +1.60 |
| historical q0 | **+0.70 pp** | +0.45 to +0.975 |
| random | **+0.30 pp** | +0.15 to +0.475 |
| q1 | **+0.425 pp** | +0.225 to +0.625 |
| q2 | **+0.425 pp** | +0.225 to +0.65 |
| q3 | **+0.45 pp** | +0.25 to +0.675 |
| q4 | **+0.45 pp** | +0.25 to +0.675 |

The equal-opponent field macro was +0.571 points with a 95% interval of +0.475 to +0.671. q1–q4
remain correlated descendants, so this is descriptive heterogeneity evidence rather than independent
replication.

Treatment's direct win rate was 50.675% [50.425%, 50.925%]. This cell is deliberately
**descriptive only** because both arms share one RNG identity inside each direct game. It did not
enter structural adoption or the practical-lift claim.

## Interpretation

The result confirms a narrow causal statement inside the declared simulator: when control and
treatment receive aligned setup, seat, and RNG streams, forcing a publicly guaranteed current-turn
win produces a small, consistently positive performance gain. The hard envelope exactly fixes the
measured failure mode and is suitable as a fixed rule component while later steps investigate
long-horizon learning.

The result does not show broad game optimality, establish strength against independent expert
opponents, or measure checkpoint-training uncertainty. The validator independently recomputed
schedules, arenas, holdout inventory, bootstrap statistics, and replay/tie/prefix counts, but it
still shares the project's engine and card rules.

## Runtime and validation

The claim runner and independent validator completed in **5 hours 9 minutes**, inside the eight-hour
step budget. The validator ran at the exact clean claim source and reproduced every retained result.
No repair or retry was used.

## Provenance

| Item | Identity |
| --- | --- |
| Claim source | `5bbda9e18d23920f5ad7686e055615aad703c097` |
| Package code fingerprint | `5dd1d77e67804cf959ccd9a90dd9569a54924623439a30194111da3304c3b4f6` |
| Plan fingerprint | `3cce3b1795043b8034916d56fa95a5db4ad1cd77477a3fc358d2115ee619b758` |
| Result fingerprint | `82bdfa96eb47febc37e2e563aa267caf95a2f18883c2b2abe7dad68e0b2f7785` |
| Independent validation | `052faad2f401a512205b6454a2c9b89b4c196f294665d073cfdb25d2d84d4b70` |
| Setup holdout | `1c8651e002d111787ca2a8413bb0e404348a960906d21c6fd6b1e3a8993621c2` |
| Archive manifest | `d6b7e36ba79cb9edc8cd959b8a07786970ab2f376a8fa80c03b69598b25f6713` |
| Archive SHA-256 | `e98ec730e4f4d007e3665f083c66edcaf1877c7318ba541af55f0b3556957c37` |

Retained evidence is under `runs/m7-terminal-offense-confirm-v1/`. The verified ignored archive is
`artifacts/archive/m7-terminal-offense-confirm-v1-2026-09-11.tar.gz` with 104 checksummed payload
files.
