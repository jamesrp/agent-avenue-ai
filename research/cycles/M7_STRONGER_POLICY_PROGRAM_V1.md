# Approved research program: stronger policy v1

**Program ID:** `m7-stronger-policy-program-v1`
**Status:** Step 4 trivial repair 2 authorized; completion pending
**Stopped:** Superseded by September 13, 2026 repair authorization
**Briefing:** [`docs/M7_STRONGER_POLICY_PROGRAM_BRIEFING.md`](../../docs/M7_STRONGER_POLICY_PROGRAM_BRIEFING.md)
**Approved:** September 11, 2026
**Parent result:** [`docs/Q0_STRENGTH_AUDIT_RESULTS.md`](../../docs/Q0_STRENGTH_AUDIT_RESULTS.md)
**Selected policy entering program:** `q0-terminal-safety-v1`

## Goal

Build and evaluate a materially stronger, information-safe Agent Avenue policy by separating five
questions in order:

1. confirm and harden the exact immediate-win tactical envelope;
2. test population replay with the current encoder/model;
3. test a structured public-history and candidate-consequence model on fixed population data;
4. add information-safe counterfactual rollout supervision; and
5. run an independent league and promotion decision.

The program should explain where strength comes from, not merely produce one favorable checkpoint.
A negative result at any intermediate step is retained and does not silently change the next step's
question.

## Why this program

The September 9 audit established that selected q0 is competent but non-optimal:

- 81.30% versus random, 68.45% versus `greedy-public-v1`, and 54.45% versus historical q0;
- only 19.20%–25.65% versus q1–q4 descendants;
- 589 misses in 1,832 publicly guaranteed current-turn wins; and
- highly predictable recruit responses, including the persistent Codebreaker-up/Saboteur-down
  pattern.

The training mechanism predicts the eventual outcome of only the action actually selected, then
maximizes over every legal action at inference. The current encoder omits chronological public
history and explicit candidate consequences. More data or more parameters alone may therefore
reinforce unsupported action values or one-opponent specialization.

## Authorized sequence

### Step 1 — terminal-offense confirmation and RNG hardening

Confirm that forcing publicly guaranteed current-turn wins is information-safe, never misses an
eligible win, and improves or preserves q0 when control and treatment use aligned random streams.
Audit exact neural max-logit ties and policy-ID sensitivity. This step may add a generic explicit RNG
identity seam to trusted runners if necessary. It does not promote or deploy a new champion.

### Step 2 — population replay with model v1

Keep the current observation encoder, candidate MLP, selected-action Monte Carlo target, optimizer,
and tactical envelope fixed. Compare q0-family-only replay with a predeclared mixed population that
includes q0, q1–q4, heuristic, random, and fixed tactical variants. Use multiple independent
corpus/training replicates and matched development arenas. This isolates collection distribution.

### Step 3 — structured model v2

On fixed step-2 population data, compare model v1 with an information-safe structured model that adds
public history, explicit candidate-consequence features, and separate play/recruit computation. A
belief or response auxiliary head may be included only if its public inputs, offline targets, loss,
and ablation are frozen before training. This isolates representation/model structure.

### Step 4 — counterfactual rollout supervision

On a frozen panel of public positions, evaluate all legal actions with one action fixed across every
information-consistent hidden-state sample. Use common rollout randomness, exact tactical outcomes, a
frozen continuation population, and no authoritative-hidden-state candidate selection. Compare the
step-3 recipe with and without joint rollout-derived listwise/value supervision.

### Step 5 — independent league and promotion

Evaluate q0, q1–q4, the tactical q0 variant, and surviving step-2/3/4 recipe replicates against a
predeclared independent league on fresh disjoint blocks. Include a stronger external
search/heuristic challenger if step 4 produces one. Use block- and replicate-aware uncertainty,
worst-opponent checks, seats, tactical invariants, and an immutable promotion decision. Only this
step may change the selected champion or web default.

## Scientific rules

- Freeze one exact step agreement and implementation revision before each claim-generating run.
- Use safe `PlayerObservation` inputs at live decision time. Hidden cards may be offline labels or
  sampled latent variables, but one public action must be shared across indistinguishable worlds.
- Keep exact guaranteed-win and avoidable-loss handling outside learned long-horizon values.
- Run at least three independent training replicates for recipe claims in steps 2–4.
- Keep controls matched within each step and change one major factor at a time.
- Retain complete semantic corpora, datasets, checkpoints, arena records, plans, decisions, logs,
  validation, reviews, and checksum archives.
- Do not reuse a block for a later locked-final claim after its result influenced design or
  selection.
- Report specialization and Pareto tradeoffs rather than forcing an Elo-like scalar when policies
  are non-transitive.
- A failed step remains evidence. Do not tune the declared recipe after inspecting its result.

## Operating limits

- Compute budget: up to **8 hours per step**, CPU-only on the current two-core VM.
- At most one substantial collection/training/evaluation job at a time.
- At most two bounded implementation/analysis workers in isolated Git worktrees at a time.
- Up to three strictly mechanical repairs per step are permitted under
  [`M7_TRIVIAL_REPAIR_POLICY_V2.md`](M7_TRIVIAL_REPAIR_POLICY_V2.md); scientific changes still
  require new explicit approval.
- Shelley remains the sole user-facing lead and integrates all work on `main`.
- Each step originally permitted at most one operational repair. This is superseded by the approved
  trivial-repair policy: up to three mechanically proven, estimand-preserving repairs; no scientific
  recipe, seed, budget, or decision change.
- The lead may proceed from one authorized step to the next without another user approval, but must
  update the committed progress record and freeze the next exact step before generating claims.
- Stop after step 5 briefing. A sixth recipe, larger budget, paid service, GPU, external credential,
  or deployment beyond the step-5 decision requires new approval.

## Program deliverables

- one exact committed agreement and compact result note per step;
- versioned information-safe tactical, population, structured-model, rollout, and league machinery;
- at least one fresh-context review per claim-generating step;
- a living progress/decision log at
  [`M7_STRONGER_POLICY_PROGRESS.md`](M7_STRONGER_POLICY_PROGRESS.md); and
- a final program briefing explaining causal evidence, failures, selected policy, and next options.
