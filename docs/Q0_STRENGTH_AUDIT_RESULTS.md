# q0 strength and tactical-leak audit: results

**Cycle:** `m7-q0-strength-audit-v1`
**Status:** Complete; selected q0 is competent but demonstrably non-optimal
**Approved and completed:** September 9, 2026
**Agreement:** [`research/cycles/M7_Q0_STRENGTH_AUDIT_V1.md`](../research/cycles/M7_Q0_STRENGTH_AUDIT_V1.md)
**Fresh-game source:** `2cefd151383e9cf3a614e2b58ee3ac2450727a61`

## 1. Bottom line

`q0-terminal-safety-v1` looks like a credible intermediate opponent, not an expert or near-optimal
one.

It remained strong against the available simple anchors on one fresh common block:

| Opponent | Games | q0 win rate | Paired-bootstrap 95% interval |
| --- | ---: | ---: | ---: |
| random | 2,000 | **81.30%** | 79.55%–82.95% |
| `greedy-public-v1` | 2,000 | **68.45%** | 66.40%–70.50% |
| historical q0 | 2,000 | **54.45%** | 52.55%–56.35% |

But it was decisively exploited by every shielded q1–q4 descendant, winning only 19.20%–25.65%.
Those descendants were previously rejected because they regressed *relative to q0* on the frozen
heuristic guardrail—not because they were weak absolutely. The selected-champion label is therefore
protocol-relative; this audit does not support treating q0 as the strongest checkpoint available in
the repository.

Most concretely, q0 missed **589 of 1,832 publicly guaranteed current-turn wins (32.15%)**. A narrow
information-safe wrapper that forced those wins converted every opportunity on its own trajectories
and improved q0 by 0.25–1.60 percentage points against each shared opponent. This is a real tactical
leak with measurable cost, but it explains only a small fraction of the q1–q4 gap.

No champion changed because this was a diagnostic cycle, not a promotion decision.

## 2. Frozen design and integrity

The audit used one common set of 1,000 setup seeds for every unordered matchup among nine fixed
policies:

- selected q0;
- q0 plus the diagnostic immediate-win wrapper;
- shielded q1–q4;
- historical q0;
- `greedy-public-v1`; and
- random.

Each setup was played with both seat assignments. The result contains 36 matchups, **72,000 games**,
and **1,015,338 decisions**. Matchup confidence intervals resample the 1,000 two-game setup blocks.
The common block improves matched contrasts but correlates matchup estimates; the tournament is not
an independently sampled policy population.

Independent validation at source `58e8e1008a66f746d7ecdcac7bba0cc28d999fe2`:

- reloaded and replayed all 72,000 semantic game records;
- exactly recomputed all 36 arena aggregates;
- verified the common setup block and every corpus/artifact fingerprint;
- exactly reproduced the tactical and recruit-pattern audits;
- confirmed zero q0 avoidable immediate-loss violations; and
- confirmed zero missed guaranteed wins by the diagnostic offense wrapper.

The first validator invocation stopped immediately because it checked the result fingerprint using
the wrong field-normalization convention. The one authorized repair changed only validator logic; it
did not alter games, seeds, policies, reports, or analysis.

## 3. Strength triangulation

### Selected q0 against the complete declared field

| Opponent | q0 win rate | 95% interval | Seat 1 | Seat 2 | Avg. score margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| q1 | **19.20%** | 17.50%–20.85% | 17.6% | 20.8% | -3.13 |
| q2 | **20.70%** | 18.95%–22.45% | 19.4% | 22.0% | -2.99 |
| q3 | **19.90%** | 18.25%–21.55% | 18.1% | 21.7% | -3.25 |
| q4 | **25.65%** | 23.90%–27.45% | 24.5% | 26.8% | -1.95 |
| historical q0 | **54.45%** | 52.55%–56.35% | 53.0% | 55.9% | +0.64 |
| heuristic | **68.45%** | 66.40%–70.50% | 68.9% | 68.0% | +3.52 |
| random | **81.30%** | 79.55%–82.95% | 78.3% | 84.3% | +4.37 |

The equal-opponent descriptive means were 68.07% over the three anchors, 21.36% over q1–q4, and
41.38% over all seven. These are summaries of this deliberately unusual field, not ratings.

### What the descendants imply

| Policy | Equal-opponent mean vs the other seven core policies | Worst point matchup |
| --- | ---: | ---: |
| q1 | 65.58% | 49.05% vs q3 |
| q3 | 65.24% | 50.95% vs q1 |
| q2 | 63.34% | 46.75% vs q1 |
| q4 | 62.11% | 48.35% vs q3 |
| heuristic | 47.57% | 31.55% vs q0 |
| selected q0 | 41.38% | 19.20% vs q1 |
| historical q0 | 37.29% | 18.40% vs q1 |
| random | 17.48% | 13.85% vs q1 |

Fresh q3 had no point-estimate loss in this core field: 84.0% versus random, 80.3% versus historical
q0, 80.1% versus selected q0, and 57.1% versus the heuristic. Its q1 and q4 comparisons remained
statistically close. This does not promote q3 or prove broad superiority: q1–q4 share one training
lineage, and q0 still outperformed q3 against the heuristic by 11.35 percentage points. It does show
that “rejected by the guardrail” must not be paraphrased as “weaker than q0.”

The fairest vacuum assessment is therefore:

- q0 reliably punishes weak and hand-authored play;
- it is somewhat better than historical q0;
- it is highly exploitable by policies trained in its own family; and
- no scalar strength estimate is credible without stronger independent opponents or humans.

## 4. Exact missed-lethal audit

A guaranteed win was defined only from the acting player's information set. For a play offer, both
opponent recruit choices had to end the current turn with the actor winning. For a recruit decision,
every face-down identity consistent with the public observation had to win. The engine's exact score,
Codebreaker, Daredevil, deck-exhaustion, simultaneous-condition, and active-player tie rules were
used.

| Phase | Opportunities | Misses | Miss rate | Miss did not win immediately | Miss eventually lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Play | 1,363 | 494 | **36.24%** | 368 | 94 |
| Recruit | 469 | 95 | **20.26%** | 76 | 25 |
| **Total** | **1,832** | **589** | **32.15%** | **444** | **119** |

The mechanisms overlap because one action can win in multiple ways:

| Mechanism present | Opportunities | Misses |
| --- | ---: | ---: |
| Score-gap threshold | 1,621 | 517 |
| Third Codebreaker | 555 | 175 |
| Opponent third Daredevil | 510 | 121 |
| Active-player tie break | 287 | 46 |

This proves q0 is not playing optimally even at zero-ply terminal tactics. It does **not** estimate
how many longer-horizon wins it misses.

The safety shield still did exactly its narrower job: across 113,108 q0 decisions it executed zero
avoidable publicly provable immediate losses. It used 4,188 all-actions-losing fallbacks. “Forced”
here means every action lost immediately under the public terminal test, not that eventual defeat
was strategically unavoidable.

### Representative misses

1. **The Sentinel/Double Agent pattern occurred.** q0 led 12–7, had three Sentinels and one Double
   Agent in its tableau, and held Sentinel, Double Agent, and Codebreaker. Offering Sentinel plus
   Double Agent in either orientation guaranteed an immediate score-gap win. q0 instead offered
   Double Agent face up and Codebreaker face down. Its model assigned logits 1.223 to the chosen
   action and 1.186 to one lethal orientation. q0 eventually won, but declined a forced finish.
2. **A missed finish became a loss.** q0 led 10–8 with two Codebreakers in its tableau and held
   Codebreaker, Sentinel, Daredevil, and Enforcer. Codebreaker plus Sentinel in either orientation
   guaranteed the current-turn win. q0 chose Enforcer face up and Daredevil face down; the selected
   logit was 2.454 versus 1.475 for one forced-win orientation. It eventually lost.

These are model-ranking errors over fully encoded legal actions, not hidden-card hindsight.

## 5. Value of the immediate-win wrapper

The diagnostic `q0-terminal-offense-v1` restricts q0 to guaranteed current-turn wins when any exist
and otherwise leaves its legal set unchanged. It converted 1,693 of 1,693 opportunities on its own
trajectories.

| Shared opponent | q0 | Wrapped q0 | Matched difference | 95% interval |
| --- | ---: | ---: | ---: | ---: |
| random | 81.30% | 81.55% | **+0.25 pp** | +0.05 to +0.50 |
| heuristic | 68.45% | 70.05% | **+1.60 pp** | +1.05 to +2.20 |
| historical q0 | 54.45% | 55.00% | **+0.55 pp** | +0.25 to +0.90 |
| q1 | 19.20% | 19.45% | **+0.25 pp** | +0.05 to +0.50 |
| q2 | 20.70% | 21.65% | **+0.95 pp** | +0.55 to +1.40 |
| q3 | 19.90% | 20.30% | **+0.40 pp** | +0.15 to +0.70 |
| q4 | 25.65% | 26.15% | **+0.50 pp** | +0.20 to +0.85 |

The wrapper beat unwrapped q0 directly at 51.0% over 2,000 games, with a paired interval of
50.6%–51.45%. The small positive effects are consistent across the declared suite, but policy IDs
use distinct deterministic RNG streams and exact max-logit tie frequency was not audited. That is a
caveat for effects measured in tenths of a point.

The evidence supports “missed guaranteed finishes have real cost.” It does not support “terminal
offense is the main q0 problem”: the wrapper still won only 19.45%–26.15% against q1–q4.

## 6. Repeatable recruit behavior

Across 57,036 q0 recruit decisions, q0's face-up choice rate was 50.8% overall but extremely
card-dependent:

| Visible card | Decisions | Took face up |
| --- | ---: | ---: |
| Saboteur | 6,712 | **0.0%** |
| Mole | 588 | 3.7% |
| Codebreaker | 11,344 | 12.4% |
| Daredevil | 7,107 | 43.8% |
| Enforcer | 9,546 | 62.0% |
| Double Agent | 12,012 | 77.6% |
| Sidekick | 640 | 85.6% |
| Sentinel | 9,087 | **95.1%** |

The old `Codebreaker face-up / Saboteur face-down` response leak persists descriptively. That ordered
offer occurred 4,609 times; q0 took the visible Codebreaker only 10.7%, its average realized immediate
score swing was -0.893 versus +0.752 for the alternative, and it won 16.7% of those games.

A second stark pattern was `Sentinel face-up / Sidekick face-down`: 529 observations, 98.5% Sentinel
selection, average realized swing -3.603 versus +3.070 for the alternative, and 17.4% game wins.
`Saboteur face-up / Mole face-down` produced another deterministic-looking trap: q0 never chose the
visible Saboteur in 843 observations, the realized chosen swing averaged -1.891 versus +1.795, and q0
won 21.9%.

These are legitimate exploit-surface clues because the offerer knows the hidden card and can choose
such pairs. They are not clean causal estimates of recruit error: q0 does not see the hidden card,
offer signaling and future tableau value matter, and the rows mix opponents and state quality.

## 7. Interpretation and next step

The main belief update is that terminal safety turned a catastrophically unsafe learned policy into a
competent baseline, but not an expert one. The next limiting problem is broader action ranking and
robustness, not merely refusing immediate losses. q0 misses exact wins, exposes highly predictable
recruit responses, and is dominated head-to-head by every descendant trained against its family.

The immediate-win wrapper is a promising low-risk engineering improvement, but deployment was outside
this cycle. Before changing the web opponent, a small approved confirmation should audit neural
max-logit ties and policy-ID permutation sensitivity.

The higher-value research step is an independently designed evaluation and training population:
multiple training replicates, opponents not descended from q0, disjoint fresh setup blocks, and a
stronger search/heuristic challenger. That would determine whether q3-like policies are genuinely
broader improvements or merely another correlated specialization, and would give q0 a more credible
absolute reference.

## 8. Provenance and retained evidence

| Item | Identity |
| --- | --- |
| Fresh-game source | `2cefd151383e9cf3a614e2b58ee3ac2450727a61` |
| Package code fingerprint | `ae8cd87f4b60c4b32a2b0ac728b1c3c0ec51f74944b9afb4d5089a1547d2e4e5` |
| Plan fingerprint | `9fa7c9acf63144aaa259d25ffd1ade6cf69e979fe9002331ddfbf395e31df61d` |
| Result fingerprint | `3369963613342f2ea3b666fe6c3cbb4dea63a0d37f94a056d625b566c4c0ebe7` |
| Independent validation | `f7101b7ffa5543d7940eb7d14997ee990c0c7e6226a4f3323e1c6c36d691619a` |
| Tactical audit | `84430987cd7c4d4dedbbbcaa74f53581e3049651df84d6acd5eddafceaf371c5` |
| Recruit-pattern audit | `f9ffd8b4d2e5f515b4c911012248d0083c08437faeb8e6915c73cf1d2043505a` |
| Archive manifest | `316d6911b48233e007f676057b13817da4c7a02a5cb0a448125326c3e8f93af7` |
| Archive SHA-256 | `06f14b7b8164b7c5ce98deff01b4e51e53a3811157ea357821600619258a52bb` |

The ignored retained root is `runs/m7-q0-strength-audit-v1/`. It contains every compressed arena
corpus, aggregate, exact trace, plan, analysis artifact, validation result, driver log, and the
fresh-context review. The verified archive is
`artifacts/archive/m7-q0-strength-audit-v1-2026-09-09.tar.gz` (24 MB compressed, 119 checksummed
payload files).
