# Milestone 7 diagnostic readiness cycle: results and briefing

**Cycle:** `m7-diagnostic-readiness-v1`  
**Status:** Complete with one declared operational repair  
**Approved:** September 8, 2026  
**Completed:** September 8, 2026  
**Agreement:** [`research/cycles/M7_DIAGNOSTIC_READINESS_V1.md`](../research/cycles/M7_DIAGNOSTIC_READINESS_V1.md)

## 1. What we learned

The main belief update is **not** “counterfactual ranking is proven to work.” The retained evidence
supports a narrower conclusion: the next controlled experiment should isolate target/action coverage
before increasing model capacity. Counterfactual candidate-ranking supervision is the preferred
first question because it can be tested on fixed retained corpora and changes one major factor more
cleanly than collecting a mixed-opponent corpus.

Confidence in that ordering is moderate at best. Mixed-opponent replay remains a close second, and
missing public-history features remain a plausible alternative explanation.

The cycle's three hypotheses resolved as follows:

1. **Partially supported:** repeated heuristic guardrail failures and poor transfer to rejected-policy
   interactions point toward target/distribution coverage rather than raw capacity, but do not
   identify one causal mechanism.
2. **Supported:** retrospective diagnostics narrowed the next experiment without producing a new
   strength claim.
3. **Supported as QA/security evidence:** historical q0 and terminal-safety q0 now work in the private
   web UI from either human seat through opaque server allowlist keys.

## 2. Results and uncertainty

### Evidence integrity

| Check | Result |
| --- | ---: |
| Whole-file archives verified | 3 |
| Archive member checksums verified | 15 q0, 137 Milestone 6, 237 terminal-safety |
| Live retained corpora semantically verified | 93 of 93 |
| Disposable restored corpora verified | 1 q0, 27 Milestone 6, 62 terminal-safety |
| Arena aggregates recomputed from retained records | 38 matched, 0 mismatched |
| Learned web checkpoint/seat cases | 4 of 4 passed |

The audit traversed 85,224 retained record instances containing 1,236,630 decisions. These are
**retained instances, not unique independent games**: the tree intentionally contains reproduction,
arena, and related corpora with overlap. They support integrity and coverage auditing, not a new
sample-size claim.

The aggregate labels are exactly balanced: 618,315 acting-player wins and 618,315 losses, with the
same count for each seat and phase. This is largely structural—each resolved turn supplies one play
and one recruit decision from opposing viewpoints—not proof that every state/action region is well
covered. Candidate roles are highly uneven: for example, Mole was face-up in only 4,839 of 618,315
play offers, while Double Agent was face-up 157,198 times. Face-up recruitment was selected 389,276
times versus 229,039 face-down selections.

### Repeated heuristic non-regression failures

| Candidate | Candidate minus incumbent vs heuristic | 95% paired-block interval | Decision |
| --- | ---: | ---: | --- |
| q1 | -0.060 | [-0.133, 0.010] | retain q0 |
| q2 | -0.113 | [-0.170, -0.055] | retain q0 |
| q3 | -0.062 | [-0.122, -0.003] | retain q0 |
| q4 | -0.120 | [-0.180, -0.058] | retain q0 |

All point estimates are negative; q2–q4 intervals are wholly below zero. This robustly establishes
the declared guardrail pattern. It does **not** establish why the regression occurred.

### Held-out all-pairs prediction diagnostic

| Model | Interaction cells | Macro chosen-action log loss | 95% interval |
| --- | ---: | ---: | ---: |
| q0 | 1 | 0.5413 | [0.5079, 0.5760] |
| q1 | 3 | 0.5447 | [0.5248, 0.5652] |
| q2 | 6 | 0.5511 | [0.5367, 0.5659] |
| q3 | 10 | 0.5582 | [0.5484, 0.5682] |
| q4 | 15 | 0.5628 | [0.5547, 0.5710] |

The numerical sequence rises, but the matchup pool expands from one to fifteen cells. These are not
same-distribution longitudinal estimates, so this is descriptive transfer evidence—not a valid
trend test. One training replicate per generation also leaves between-training variability
unmeasured.

Supporting artifacts:

- `runs/research-cycles/m7-diagnostic-readiness-v1-repair1/analysis/analysis.json`
- `runs/research-cycles/m7-diagnostic-readiness-v1-repair1/analysis/tables.md`
- `runs/research-cycles/m7-diagnostic-readiness-v1-repair1/analysis/diagnostic-signals.svg`
- `runs/research-cycles/m7-diagnostic-readiness-v1-repair1/raw/artifact-catalog.json`
- `runs/research-cycles/m7-diagnostic-readiness-v1-repair1/raw/diagnostic-summary.json`

## 3. Failures, repairs, and deviations

The first frozen run at source `71ad8be` verified archives and corpora but reported zero arena
recomputations. The cause was an implementation defect: the audit recognized direct CLI arena
reports but not the wrapped arena artifacts emitted by iteration runs. Dependent interpretation was
stopped rather than accepted.

The single preapproved operational repair added wrapped-report support and required nonzero matched
recomputation evidence. The repair was committed as `b614646`, frozen in repair plan
`research/cycles/m7-diagnostic-readiness-v1-repair1.json`, and executed at source `f9c6cd7`. It
recomputed 38 reports with no mismatch. Final conclusions use only the repaired run.

Delegation also had one operational failure: the web worker returned no artifact after its one
follow-up. That did not block independent work. The lead implemented and tested the authorized web
package while the diagnostics worker completed its separate assignment. The diagnostics worker's
first full live scan exceeded its 15-minute tool-call cap; the managed frozen run completed normally
in about 37 minutes.

The original run plus repair, review, and briefing remained well inside the eight-hour cycle budget.
No optional 100-pair confirmation arena was needed. No training, promotion, or locked-final seeds
were opened.

## 4. Scientific limitations and alternatives

- The retained data contains only outcomes for selected actions. It does not reveal what would have
  happened after every legal alternative.
- Rising crossplay log loss is confounded by the expanding interaction pool.
- Each generation has one training replicate, so corpus/training variability is unknown.
- The terminal-safety q0 comparison combines retraining and deployment shielding; it does not isolate
  the shield's causal contribution.
- Heuristic guardrail regression may represent a policy tradeoff rather than globally weaker play.
- Public action history is absent from the 87-feature encoder and may limit belief-sensitive choices.
- Aggregate coverage counts can conceal rare state/action combinations; exact target balance does
  not imply useful counterfactual coverage.
- Safe traces validate the retained `PlayerObservation` construction examined here, not every future
  UI or agent implementation.
- Archives have checksum-verified same-VM copies, but no off-VM durability.

The independent reviewer rated archive integrity and the repeated guardrail pattern high-confidence,
selected-action/opponent coverage as moderately important, and ranking-over-mixed-replay ordering as
low-to-moderate confidence. Review: `runs/research-cycles/m7-diagnostic-readiness-v1-repair1/fresh-context-review.md`.

## 5. Board-game ML explanation: selected-action support

The current model learns:

```text
observation + action actually chosen -> eventual game outcome
```

That label says whether the chosen action was followed by a win. It does not say whether another
legal action would have been better. If self-play repeatedly chooses a narrow subset, the model gets
many labels for familiar actions and little direct ordering information for alternatives.

Counterfactual ranking supervision would add a teacher's relative preference across several legal
actions in the **same public position**. This may teach useful early rankings and is experimentally
clean because retained positions can be reused. The danger is imitation bias: a heuristic teacher
can transfer its blind spots or cap the model near heuristic behavior. The proposed experiment must
therefore treat ranking as a warm start or auxiliary target and judge success only through fresh
paired gameplay.

Mixed-opponent replay attacks a related problem from the data-distribution side. It may produce more
diverse states, but it changes opponent composition and collection behavior together, making a
negative or positive result harder to attribute.

## 6. Recommended next decisions

### Recommendation

Design—but do not yet execute—a controlled **counterfactual candidate-ranking supervision**
experiment on fixed retained corpora. Compare it with the unchanged selected-action Monte Carlo
recipe using matched arenas and multiple independent training seeds. This is recommended as the
most directly isolating next question, not as an evidenced solution.

### Tradeoffs requiring user direction

| Choice | Benefit | Main risk |
| --- | --- | --- |
| Ranking supervision first | Isolates target/action-ordering signal on fixed data; lower collection cost | Imitates heuristic bias; may not fix distribution shift |
| Mixed-opponent replay first | Directly broadens policy-interaction coverage | Changes collection distribution and opponent pool together |
| Public-history encoder first | Tests a plausible missing-information representation | Adds representation complexity before target coverage is resolved |
| Off-VM artifact retention | Protects accumulated scientific evidence | Requires selecting/configuring external storage |

No next research cycle has been started. The user must choose whether to authorize a ranking-
supervision experiment design/execution, another diagnostic direction, or artifact-infrastructure
work.

## Appendix: provenance and inspectable evidence

- Approved agreement revision: `73afa61`
- Original frozen source: `71ad8be`
- Repaired frozen source: `f9c6cd7`
- Original plan fingerprint: `14b4162b25eaa18df43c5a5d36a296658ee0634b2b1254ccc0644dde2a18907f`
- Repair plan fingerprint: `e916cb150df766497f7e952d5db23dd54badfcb7ff861e94ee7210c810af21a1`
- Original runtime: `runs/research-cycles/m7-diagnostic-readiness-v1/`
- Final runtime: `runs/research-cycles/m7-diagnostic-readiness-v1-repair1/`
- Hand-checkable positions: `.../analysis/safe-traces.md`
- Learned web evidence: `.../web/actual-checkpoints.json`
- Reviewer report: `.../fresh-context-review.md`
