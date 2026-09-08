# M7 heuristic candidate-ranking warm-start: results and briefing

**Cycle:** `m7-heuristic-ranking-warmstart-v1`  
**Final runtime:** `m7-heuristic-ranking-warmstart-v1-repair1`  
**Status:** Complete; frozen recipe does not advance  
**Approved and completed:** September 8, 2026  
**Agreement:** [`research/cycles/M7_HEURISTIC_RANKING_WARMSTART_V1.md`](../research/cycles/M7_HEURISTIC_RANKING_WARMSTART_V1.md)

## 1. Decision

The exact frozen ranking warm start **does not advance**. It preserved substantial absolute strength
against shielded q0 and random, but it did not beat its matched MC-only controls and it made the
heuristic regression materially worse.

The machine artifact reports `inconclusive` because the implementation encoded only `advance` versus
non-advance and no numerical rejection branch was predeclared. The practical research decision is
clearer: **reject this exact recipe as an advancement candidate**. This does not establish that all
ranking supervision is harmful.

No champion changed. `q0-terminal-safety-v1` remains selected, and no locked-final seeds were opened.

## 2. Frozen advancement results

| Criterion | Result | Required | Pass? |
| --- | ---: | ---: | --- |
| Treatment vs matched MC control | 48.37% [45.67%, 51.10%] | lower endpoint > 50%; at least 2/3 replicate points > 50% | No |
| Treatment vs shielded q0 parent | 75.77% [73.47%, 78.03%] | lower endpoint > 50% | Yes |
| Heuristic difference vs parent | -18.17 pp [-24.17, -12.28] | lower endpoint > -5 pp | No |
| Treatment vs random | 71.33% [67.50%, 75.17%] | lower endpoint > 50% | Yes |
| Minimum candidate seat rate | 39.0% | at least 45% | No |
| Immediate-loss safety | 0 executed avoidable provable losses | zero | Yes |

Intervals are deterministic nested 95% bootstrap intervals. They resample the three training
replicates and the shared paired setup blocks rather than pooling every game as one fixed model.

Direct treatment-versus-control replicate point estimates were 48.5%, 47.1%, and 49.5%; none favored
the treatment. The 39.0% seat failure occurred for treatment replicate 3 as Player One against the
heuristic.

## 3. Per-replicate gameplay

| Replicate | Treatment vs control | Treatment vs parent | Control vs parent | Treatment vs heuristic | Control vs heuristic | Treatment vs random | Control vs random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 48.5% | 75.7% | 81.9% | 49.5% | 55.83% | 72.25% | 77.75% |
| 2 | 47.1% | 75.9% | 82.2% | 46.0% | 56.17% | 71.75% | 79.75% |
| 3 | 49.5% | 75.7% | 82.3% | 42.5% | 56.33% | 70.0% | 80.25% |

The aligned parent reference won 64.17% against the heuristic on these development blocks. The
ranking treatments therefore failed the original robustness problem by a wide margin despite still
beating the parent head to head.

Exploratory treatment-minus-control contrasts, which were **not** advancement statistics, were also
negative against every external opponent: -6.37 pp versus parent, -10.11 pp versus heuristic, and
-7.92 pp versus random. Their post-hoc nested intervals were wholly below zero. These contrasts are
descriptive and cannot replace the frozen criteria.

## 4. Ranking coverage and training behavior

The all-legal-action dataset reconstructed the same 61,712 public decision positions from the fixed
4,000-game q1 corpus:

| Coverage item | Count |
| --- | ---: |
| Public positions | 61,712 |
| Encoded legal candidates | 294,639 |
| Strict teacher preference pairs | 832,505 |
| Tied candidate pairs | 146,322 |
| Tie-only positions | 3,942 |
| Train positions / pairs | 55,282 / 743,067 |
| Validation positions / pairs | 6,430 / 89,438 |

The ranking initializers reached 93.36%–93.40% validation pair accuracy, so the pretraining stage did
learn the teacher's ordinal labels. That did not translate into better final gameplay after MC
fine-tuning.

| Replicate | Ranking best epoch | Control MC best epoch | Treatment MC best epoch | Control equal-game validation BCE | Treatment equal-game validation BCE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 49 | 33 | 49 | 0.53130 | 0.53567 |
| 2 | 50 | 44 | 49 | 0.53061 | 0.54094 |
| 3 | 50 | 38 | 50 | 0.53031 | 0.54015 |

All treatment MC fits reached the 50-epoch limit and had worse held-out BCE and Brier score than their
controls. This is consistent with a harmful or persistent initialization bias, but the run did not
retain ranking-only gameplay or intermediate MC checkpoints, so it cannot distinguish persistence
from washout dynamics.

Replicate 1's control reproduced the historical q1 tensor digest exactly:
`14d6f9b2c3fbacf0100f497df48a87ec27594e6ff5c058549a34dafe72b07f48`.

## 5. Evidence validation

- 22 arena artifacts and all 22 retained arena corpora were independently reloaded and recomputed.
- All aggregate reports, corpus fingerprints, artifact fingerprints, and safety diagnostics matched.
- The arenas contain 15,600 games and 239,362 decisions.
- There were zero executed **avoidable** provable losses. The shield used 7,418 forced-loss
  fallbacks; those must not be described as zero total provable-loss actions.
- The four development domains contain 1,500 unique setup seeds: 500 direct, 500 parent, 300
  heuristic, and 200 random. They are mutually disjoint and have zero overlap with the 4,000 training
  setups or 182 other retained corpus files scanned.
- Ranking dataset arrays, six final checkpoints, three ranking initializers, nested intervals,
  advancement conditions, and result fingerprints all revalidated.
- The original failed run's replicate-1 control, ranking initializer, and treatment tensor digests
  exactly match the repaired run.

Validation artifacts:

- `runs/research-cycles/m7-heuristic-ranking-warmstart-v1-repair1/artifact-validation.json`
- `runs/research-cycles/m7-heuristic-ranking-warmstart-v1-repair1/arena-validation-00.json` through
  `arena-validation-03.json`
- `runs/research-cycles/m7-heuristic-ranking-warmstart-v1-repair1/seed-audit.json`
- `runs/research-cycles/m7-heuristic-ranking-warmstart-v1-repair1/repair-validation.json`
- `runs/research-cycles/m7-heuristic-ranking-warmstart-v1-repair1/fresh-context-review.md`

## 6. Operational failure and repair

The original frozen run at source `eb513cc` produced and validated the ranking dataset and replicate-1
models, then failed while reloading its ranking initializer. A `Tensor` type imported only under
`TYPE_CHECKING` was evaluated at runtime by `typing.cast`, causing a `NameError`. The automatic retry
repeated the same defect and stopped.

The authorized operational repair imported `Tensor` locally in the optional RL load path and added a
regression round-trip test. The repair was committed as `2de410b`, frozen in repair plan
`research/cycles/m7-heuristic-ranking-warmstart-v1-repair1.json`, and executed at source `240d334`.
It did not change the corpus, teacher, ranking loss, recipes, seeds, arena sizes, or decision criteria.
The repaired workflow completed in about 55 minutes. Final conclusions use only the repaired run.

## 7. Interpretation and limitations

### What the result supports

On this exact corpus, architecture, optimizer, shield, and development suite, heuristic pairwise
ranking pretraining followed by unchanged MC fine-tuning failed its gate. It preserved absolute
parent/random strength but underperformed matched MC-only controls on the direct comparison and on
exploratory external-opponent contrasts, while failing heuristic and seat robustness criteria on
this development suite.

### What it does not support

- Ranking supervision is generally inferior.
- The heuristic teacher caused the regression.
- Ranking signal was completely preserved or completely washed out by MC fine-tuning.
- The result generalizes to a fresh corpus, another teacher, an auxiliary loss, or shield-aware
  candidate filtering.
- Three training replicates on one corpus characterize the full population of training runs.
- A ranking treatment should be promoted or tested on locked-final seeds.

Plausible mechanisms include teacher-induced bias, ordinal-margin versus calibrated-value mismatch,
an unfavorable optimization basin, overfitting on the fixed corpus, ranking/MC seed interactions, or
spending capacity on actions that the deployment shield would veto. The current evidence does not
separate them.

## 8. Board-game ML lesson

More labels are not automatically better labels. The treatment enumerated every legal candidate and
expanded supervision from one chosen action to strict preference pairs among teacher-unequal
actions; tied pairs and tie-only positions supplied no pairwise signal. It achieved high validation
pair accuracy, yet the final policy was less successful on the frozen development gate. In an
action-value model, teacher ranking margins and terminal win probabilities impose different geometry
on the same scalar output. A warm start can therefore increase action-ordering supervision while
making later value fitting harder.

## 9. Recommended next bounded decision

Do not collect a fresh ranking corpus or tune this recipe yet. The next proposed cycle is a
**diagnostic-only teacher-retention/washout audit using existing artifacts**:

- compare parent, ranking initializer, MC control, and final treatment on a frozen safe public-state
  panel;
- measure teacher top-action and pairwise agreement, logit spread, phase/card/action changes,
  selected-action calibration, and terminal-safety veto status; and
- determine whether ranking signal persists, disappears, or is concentrated in shield-vetoed or
  rare action regions.

That cycle would require no new training or gameplay and would stop before selecting another recipe.
Mixed-opponent replay remains the next substantive data intervention if diagnostics do not justify a
narrower target repair.

No follow-up cycle has been started.

## Appendix: provenance

- Agreement commit: `7832257`
- Ranking implementation: `46efd79`
- Original frozen plan source: `eb513cc`
- Repair implementation: `2de410b`
- Final repaired source: `240d33495c29e227c57afe2b58d73ad792fbdbd8`
- Workflow repair-plan fingerprint: `05470c56456532928d3d6b60f23b8b8c8a5c93804969379a3d8b1e853a20acf9`
- Internal experiment-plan fingerprint: `2210a4304a4c706cb3936ffa6901beafa2b23adacafa6bf704b9256f9f97ead6`
- Ranking dataset fingerprint: `2bd038999a1b5f4af67a839268035dfadf27503707e0817b530b369779f2ddc7`
- Result fingerprint: `90abf5b95248ef02f891c1773d2399c2ba09afe56529c6d771d9782d81b9c012`
- Final runtime: `runs/research-cycles/m7-heuristic-ranking-warmstart-v1-repair1/`
- Archive: `artifacts/archive/m7-heuristic-ranking-warmstart-v1-repair1-2026-09-08.tar.gz`
- Archive SHA-256: `2bd65ddf8aa339d718f64cbad95a3d5349f8cfcddfda59d5fc2c87828b735730`
