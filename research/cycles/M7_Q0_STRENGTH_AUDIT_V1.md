# Approved research-cycle agreement: q0 strength and tactical-leak audit

**Cycle ID:** `m7-q0-strength-audit-v1`
**Status:** Complete
**Approved and completed:** September 9, 2026
**Result:** [`docs/Q0_STRENGTH_AUDIT_RESULTS.md`](../../docs/Q0_STRENGTH_AUDIT_RESULTS.md)
**Selected policy:** `q0-terminal-safety-v1`
**Parent evidence:** [`docs/TERMINAL_SAFETY_RESULTS.md`](../../docs/TERMINAL_SAFETY_RESULTS.md)

## Research question

How strong does the selected terminal-safety q0 policy look without a grandmaster reference, and can
fresh play reveal specific, repeatable weaknesses—especially failures to take a publicly guaranteed
immediate win?

This is a diagnostic strength audit, not a promotion cycle. It triangulates q0 against simple
anchors, historical q0, and the four known specialist descendants, then measures exact tactical
conversion and repeated offer/recruit behavior.

## Hypotheses

1. Shielded q0 will remain clearly stronger than random, `greedy-public-v1`, and historical q0, but
   will look substantially less robust against q1–q4 specialist descendants.
2. The terminal-safety shield prevents avoidable immediate losses but does not force immediate wins;
   q0 will therefore miss a measurable fraction of information-safe current-turn wins.
3. A narrow information-safe immediate-win wrapper will convert every such opportunity and improve
   q0 on at least some matched opponents, establishing the practical cost of this tactical leak.
4. Repeated q0 recruit tendencies will expose offer patterns that are useful for human inspection,
   while game-outcome correlations from those patterns will remain descriptive rather than causal.

## Frozen policy field

Evaluate these nine fixed policies:

- selected `q0-terminal-safety-v1`;
- diagnostic `q0-terminal-offense-v1`, which is the selected policy wrapped only to restrict the base
  policy to publicly guaranteed current-turn wins when any exist;
- shielded q1, q2, q3, and q4 rejected descendants from `runs/terminal-safety-v1`;
- unshielded historical q0;
- `greedy-public-v1`; and
- random.

The immediate-win wrapper is a diagnostic intervention, not a candidate champion. If no guaranteed
win exists it supplies the unchanged legal set to q0. If one or more exist it lets q0 choose only
among those actions. It must use only `PlayerObservation`, public card definitions, semantic legal
actions, and exact engine terminal adjudication.

## Exact tactical definitions

A **publicly guaranteed current-turn win** is information-set safe:

- For a play offer, both opponent recruit choices must end the current turn with the offerer winning.
- For a recruit action, every face-down identity consistent with the recruiter's public observation
  must end the current turn with the recruiter winning. A win that exists only for the realized
  hidden card is hindsight-only and is not a missed lethal.
- Score-gap, third Codebreaker, opponent third Daredevil, deck exhaustion, simultaneous conditions,
  and active-player tie resolution use the engine's exact terminal evaluator.

A **missed lethal** occurs when at least one publicly guaranteed win is legal but the recorded action
is not in that set. Report separately whether the chosen line nevertheless won immediately and
whether the player ultimately won the game.

## Frozen fresh evaluation

Use root seed `2026090901` and one common block of **1,000 paired setup seeds** for every unordered
policy matchup. Nine policies produce 36 matchups, 2,000 seat-swapped games per matchup, and **72,000
fresh games** total. Agent RNG identities remain stable by policy across matchups.

Retain compressed semantic records and report:

- the complete all-pairs win-rate matrix with paired-bootstrap 95% intervals;
- q0's anchor average, specialist-descendant average, field average, worst matchup, seat rates,
  margins, terminal reasons, and game length;
- immediate-win opportunities, actions, conversions, misses, non-immediate misses, and eventual-loss
  misses by agent, phase, opponent, and terminal mechanism;
- `q0-terminal-offense-v1` versus q0 directly, plus matched paired-block win-rate differences between
  the two policies against every shared opponent;
- exact representative missed-lethal traces and q0 candidate logits reconstructed solely from the
  safe observation;
- q0 recruit choice rates by visible card, visible-copy count, opponent, and true ordered offer pair;
  and
- all ordered offer-pair rows with at least 100 q0 recruit observations, using fixed columns for
  support, face-up choice rate, immediate assignment, and final game result.

Paired uncertainty resamples the common 1,000 setup blocks. Field averages are descriptive equal-
opponent summaries; they are not an Elo rating and must not erase non-transitivity.

## Integrity and completion criteria

- Freeze a clean source revision before fresh games.
- Validate every checkpoint, corpus manifest, game record, replay, schedule, rules fingerprint, code
  fingerprint, and all-pairs common setup block.
- The diagnostic offense wrapper must execute zero missed publicly guaranteed wins in retained games.
- The selected shielded q0 must execute zero avoidable publicly provable immediate losses.
- Analysis must be reproducible from retained records and emit a fingerprinted machine-readable
  result plus a compact committed briefing.
- Obtain one fresh-context interpretation review before the final briefing.

## Limits

- Wall-clock operating budget: **8 hours** after fresh-game dispatch.
- CPU-only current two-core VM; one substantial arena/analysis process at a time.
- At most two bounded implementation workers and one fresh-context reviewer.
- At most one operational repair and retry. A repair may fix code, resume, or reporting defects but
  may not change the nine policies, root seed, common block, pair count, tactical definitions, or
  reported metrics.
- No adaptive extra games or post-result opponent construction. Exploratory slices may use only the
  frozen 72,000-game corpus and must be labeled post hoc.
- Generated records and reports remain ignored under `runs/m7-q0-strength-audit-v1/`; compact source,
  agreement, and final briefing are committed.

## Explicitly out of scope

- claims of optimal play, solved-game value, or grandmaster equivalence;
- training, promotion, checkpoint modification, or web-opponent replacement;
- multi-turn search, omniscient hidden-card evaluation, or a general hand-authored strategy layer;
- treating q1–q4 as globally strong rather than known q0-specialized descendants; and
- launching a follow-up training or policy-change cycle.

## Decisions reserved for the user

Whether to deploy an immediate-win wrapper, build deeper search, train against a broader opponent
mixture, replace the selected champion, or open another strength cycle requires new user approval.
The cycle stops after the briefing.
