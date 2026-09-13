# Approved trivial operational-repair policy v2

**Program:** `m7-stronger-policy-program-v1`
**Status:** Approved by user
**Approved:** September 13, 2026

## Purpose

Allow more than one genuinely trivial operational repair when a frozen experiment is blocked by
packaging, serialization, orchestration, validator, or resumability mechanics that cannot alter its
scientific result.

The prior one-repair rule was intentionally conservative but treated a one-ULP metadata comparison
like a recipe change. This policy separates mechanical repairs from scientific changes.

## Allowed trivial repairs

A repair is trivial only when all are true:

1. policies, checkpoints, model inputs, targets, labels, losses, optimizer settings, seeds, sample
   counts, arena schedules, statistics, thresholds, and decision rules are unchanged;
2. no result-dependent recipe choice or additional sample is introduced;
3. existing deterministic payloads are preserved byte-for-byte or regenerated from the frozen plan
   and shown equivalent before publication;
4. the changed paths and artifact operations are explicitly allowlisted;
5. a repair ledger records the failure, source, commands, inputs, outputs, and equivalence evidence;
6. validation recomputes the frozen scientific artifacts after repair; and
7. a fresh-context review agrees that the change cannot affect the scientific estimand.

Examples include correcting shard granularity, restoring a mutable log excluded from scientific
scope, deterministic RNG replay order in a validator, integer-versus-derived fraction comparison,
and crash-safe publication.

## Limits

- Up to **three trivial operational repairs per step** may be executed without another user check-in.
- One failed trivial repair does not consume or modify the scientific recipe.
- Each repair must be committed or durably recorded before the next scientific stage resumes.
- If a proposed change can affect any model tensor, target value, game trajectory, evaluation sample,
  bootstrap sample, gate, or interpretation, it is not trivial and requires new explicit approval.
- After three trivial repairs, stop and return to the user even if every remaining issue appears
  mechanical.
- Compute remains bounded by the applicable step's approved eight-hour scientific budget; small
  repair verification overhead is recorded separately and may not justify changing sample sizes.

## Program interaction

This policy supersedes the one-trivial-repair wording in the stronger-policy program and its Step-4
agreement. It does not authorize a sixth research step or any change to the final promotion rule.
