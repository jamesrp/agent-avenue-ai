# Approved step agreement: terminal-offense confirmation v1

**Program:** `m7-stronger-policy-program-v1`
**Step:** 1 of 5
**Cycle ID:** `m7-terminal-offense-confirm-v1`
**Status:** Complete; structurally adopted for steps 2–4
**Approved and completed:** September 11, 2026
**Result:** [`docs/M7_TERMINAL_OFFENSE_CONFIRM_RESULTS.md`](../../docs/M7_TERMINAL_OFFENSE_CONFIRM_RESULTS.md)
**Parent evidence:** [`docs/Q0_STRENGTH_AUDIT_RESULTS.md`](../../docs/Q0_STRENGTH_AUDIT_RESULTS.md)

## Question

Does the information-safe immediate-win envelope satisfy its exact structural contract, and what is
its practical effect on q0 after removing policy-label-dependent RNG streams?

This step confirms one hard tactical rule. It does not train, promote, change the web default, or
answer broader model-strength questions.

## Fixed policies

- **Control:** `TerminalSafety(q0-terminal-safety-v1)` using checkpoint
  `b511ae162b794da6450fd3151762d98b4e5c8475c1af23e5b9289d0140a0a06d`.
- **Treatment:** `TerminalOffense(TerminalSafety(q0-terminal-safety-v1))` using the same checkpoint.
- **Shared opponents:** shielded q1–q4 candidates, unshielded historical q0,
  `greedy-public-v1`, and random.
- Control and treatment have distinct recorded policy IDs but share normalized RNG identity
  `q0-terminal-core-v1`. Each opponent keeps one stable RNG identity.

The treatment restricts the base policy to publicly guaranteed current-turn wins when at least one
exists. Otherwise it supplies the unchanged terminal-safety action set. The public forced-win
definition remains:

- play: both opponent recruit choices end the current turn with the actor winning;
- recruit: every face-down card identity consistent with the recruiter's public information ends the
  current turn with the actor winning; and
- exact score-gap, Codebreaker, Daredevil, deck-exhaustion, simultaneous-condition, and active-player
  tie adjudication.

## Fresh schedule

- Root seed: `2026091101`.
- Seed families: `seed-family-a` and `seed-family-b`, derived by named domains.
- Paired setup blocks per family: **1,000**.
- The two families are disjoint; every cell within a family reuses its same 1,000 setup blocks.
- Fifteen cells per family:
  - one treatment-versus-control direct arena;
  - one control arena versus each of seven shared opponents; and
  - one treatment arena versus each of seven shared opponents.
- Every setup is played with both seat assignments.
- Total: **30 cells and 60,000 games**.

Before games begin, recursively scan every completed corpus discoverable under `runs/`, excluding the
current output subtree, and require zero normalized game-config/setup-seed overlap. Retain the exact
corpus inventory and holdout fingerprint. Previously observed September 9 strength-audit blocks are
therefore excluded automatically.

## RNG and prefix contract

The runner separates policy identity from RNG identity. In each shared-opponent control/treatment
comparison, setup seed, q0-family seed/domain, opponent seed/domain, and seat are identical.

For every matched same-setup/same-seat game, semantic histories must match until the first control
decision where a guaranteed win exists. At that endpoint:

- if control misses, treatment must choose a guaranteed win;
- if control already chooses a guaranteed win, treatment may choose the same or another guaranteed
  win; and
- any earlier action divergence, metadata mismatch, or treatment endpoint failure blocks structural
  adoption.

The direct treatment-versus-control arena intentionally gives both arms the shared q0 RNG identity
inside one game. It is **descriptive only** and is not used for structural adoption or practical-lift
claims.

## Exact audits

Replay every decision and retain:

1. independent public forced-win oracle versus production classifier agreement;
2. engine-transition cross-checks for selected and guaranteed play offers;
3. control/treatment guaranteed-win opportunities, conversions, misses, and false wins by phase and
   opponent;
4. terminal-safety avoidable-loss violations;
5. raw learned candidate logits in semantic order, exact maximum ties, forced/non-forced maximum
   composition, and selected-action membership in the maximum set; and
6. matched control/treatment prefix evidence.

The independent oracle must not call the production offense classifier. Recruit uncertainty is
resolved over public-consistent identities, never the authoritative hidden card.

## Statistics

Use a deterministic 20,000-resample bootstrap stratified by seed family. Resample common setup
blocks jointly across opponents so cross-cell correlation is preserved.

Report:

- treatment-minus-control win rate for each shared opponent;
- equal-opponent **anchor macro** over historical q0, heuristic, and random;
- equal-opponent **field macro** over all seven shared opponents; and
- direct treatment win rate, labeled descriptive-only.

The practical-lift claim passes only when the anchor-macro 95% lower endpoint exceeds **+0.25
percentage points**. Individual opponent intervals and the field macro remain descriptive
heterogeneity evidence.

## Decisions

### Structural adoption for steps 2–4

Adopt the hard terminal-offense envelope as a fixed component only if all hold:

1. clean frozen source and exact checkpoint/config identities;
2. disjoint seed families and zero overlap with the exhaustive prior-corpus inventory;
3. record, replay, schedule, seat, and RNG metadata integrity;
4. zero independent-production oracle disagreements;
5. zero treatment guaranteed-win misses and false wins;
6. zero avoidable immediate-loss violations for both arms;
7. every learned selected action belongs to its exact maximum-logit set; and
8. zero matched-prefix metadata/action failures before the guaranteed-win endpoint.

Structural adoption does not require a statistically significant win-rate gain because a correctly
classified current-turn terminal win is weakly dominant by definition. If structural adoption fails
after the one allowed operational repair, later steps stop as blocked and the user receives a
briefing.

### Empirical claim

If structural adoption passes but the practical-lift criterion fails, report that the rule is exact
but the measured gain is smaller or uncertain. Continue to step 2 with the envelope fixed, without
calling it measurably stronger.

## Validation and limits

- Claim-generating source must be tracked-clean and include this committed agreement.
- The deterministic validator runs before any post-result source change and requires Git revision,
  `uv.lock`, rules fingerprint, and package code fingerprint to match the plan.
- Validation independently recomputes schedules, reports, common-block statistics, holdout inventory,
  tactical/tie counts, and prefix results rather than calling the claim runner's aggregate/audit
  functions.
- Claim-run cutoff: **7 hours 45 minutes**. An incomplete run emits no result.
- Whole-step compute budget: **8 hours**, CPU-only, one substantial process at a time.
- At most one resume/operational repair. It may not change policies, checkpoints, roots, seeds, pair
  counts, statistics, or criteria.
- Generated evidence belongs under `runs/m7-terminal-offense-confirm-v1/`; compact agreement/result
  notes and source are committed. A checksum archive and fresh-context review are required before
  step completion.

## Out of scope

- retraining or model selection;
- deeper-than-current-turn tactics;
- changing q0's selected-champion or web status;
- treating q1–q4 as independent expert opponents; and
- using the direct common-RNG arena as a causal estimate.
