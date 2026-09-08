# Approved research-cycle agreement: M7 heuristic ranking warm start

**Cycle ID:** `m7-heuristic-ranking-warmstart-v1`  
**Status:** Approved by user  
**Approved:** September 8, 2026  
**Parent evidence:** [`docs/M7_DIAGNOSTIC_READINESS_RESULTS.md`](../../docs/M7_DIAGNOSTIC_READINESS_RESULTS.md)  
**Fixed training source:** retained `runs/terminal-safety-v1/q1-a1` corpus, split, and shielded-q0 parent

## Research question

On the exact retained terminal-safety q1 corpus, does an information-safe heuristic
candidate-ranking warm start improve the subsequent selected-action Monte Carlo candidate relative
to an otherwise identical MC-only control?

This is a controlled recipe screen. The teacher supplies ordinal preferences among legal alternatives,
not observed counterfactual returns. A positive result may justify a later fresh-corpus confirmation;
it does not promote a champion or establish that heuristic ranking is generally stronger.

## Hypotheses

1. Ranking pretraining over every legal action in each retained public position will improve the
   final candidate relative to an MC-only control trained on only the selected actions.
2. The improvement, if real, will preserve the large q1 parent head-to-head gain while avoiding the
   previously observed heuristic non-regression failure.
3. Any heuristic-opponent improvement may be teacher imitation rather than general strategic
   improvement, so direct treatment/control gameplay and parent/random guardrails are required.

## Frozen recipes

Both recipes use:

- parent checkpoint `runs/terminal-safety-v1/q0-a1/checkpoint`;
- retained corpus `runs/terminal-safety-v1/q1-a1/corpus`;
- the existing q1 deterministic game split and selected-action dataset;
- encoder `candidate-public-v1`, model `candidate-mlp-v1`, and unchanged MC optimizer/training
  defaults;
- parent initialization before any recipe-specific training; and
- the same terminal-safety wrapper in every gameplay arena.

The **MC control** applies the existing q1 selected-action Monte Carlo training directly to the
parent weights. The **ranking treatment** first applies one frozen heuristic-ranking pretraining
stage to the same parent weights, then applies exactly the same MC stage as its paired control.

For each acting-player `PlayerObservation`, ranking extraction enumerates every legal semantic action
and calls the frozen `greedy-public-v1` public scorer. The ranking loss is pairwise logistic loss over
strict teacher preferences, averaged within a position before positions are averaged. Exact teacher
ties create no preference pair and are counted. Teacher score magnitude is not fitted. No hidden
card, opposing hand, deck order, or authoritative state may enter the ranking dataset or trainer.

Ranking pretraining uses deterministic CPU AdamW with the baseline learning rate, weight decay,
maximum 50 epochs, patience 8, and validation selection by equal-game-weighted ranking loss. The MC
stage retains the baseline batch size, learning rate, weight decay, maximum 50 epochs, patience 8,
and equal-game-weighted validation BCE selection.

## Replicates and seeds

Run three paired training replicates per recipe. Replicate 1 uses the retained q1 MC training seed
`3768947516520230646`; its MC control must reproduce the retained q1 tensor digest exactly before
claim-generating training continues. Replicates 2 and 3 use distinct seeds derived from cycle root
`2026090801`. Each treatment/control pair shares its MC shuffle seed. Ranking-stage shuffle seeds and
all arena/bootstrap seeds use separate named derivation domains.

The corpus, split, model architecture, parent, MC target, MC optimizer, MC seed, and deployment shield
are matched within each pair. The ranking stage is the sole major changed factor.

## Evaluation

Use only fresh declared development seed domains, never prior development or locked-final blocks.
For each replicate run:

- 500 paired treatment-versus-control blocks;
- 500 aligned blocks for each candidate versus shielded q0;
- 300 aligned blocks for each candidate versus `greedy-public-v1`, plus one matched shielded-q0
  reference arena;
- 200 paired blocks for each candidate versus random; and
- seat, score-margin, terminal-reason, safety-veto, replay, throughput, and compatibility checks.

Report every replicate separately. Aggregate uncertainty must use a deterministic nested bootstrap
that resamples the three training-replicate pairs and paired gameplay blocks; gameplay blocks may not
be pooled as if all checkpoints were one fixed model.

## Advancement rule

The ranking recipe advances only if all of the following hold:

1. treatment beats its paired MC control in at least two of three replicate point estimates and the
   nested 95% lower endpoint for mean treatment score exceeds 0.50;
2. the aggregate nested 95% lower endpoint versus shielded q0 exceeds 0.50;
3. the aligned heuristic non-regression interval versus shielded q0 has lower endpoint greater than
   -0.05;
4. the aggregate random-opponent lower endpoint exceeds 0.50;
5. no candidate seat point estimate is below 0.45; and
6. information-safety, compatibility, replay, and immediate-loss guardrails pass.

Failure to satisfy every condition is not promotion. The conclusion is `advance`, `reject`, or
`inconclusive` under the frozen criteria. No locked-final block is opened in this cycle.

## Work packages

### A. Ranking recipe implementation

- Add versioned deterministic all-legal-action ranking extraction, materialization, training,
  checkpoint lineage, and metrics.
- Add a bounded experiment runner and compact machine-readable plan/result artifacts.
- Test tie handling, equal-position/equal-game weighting, deterministic seeds, hidden-information
  safety, checkpoint compatibility, resume behavior, and unchanged MC control behavior.

### B. Fixed-input audit and control reproduction

- Verify exact corpus, dataset, parent checkpoint, source code, and rules fingerprints.
- Materialize and audit ranking coverage, preference-pair counts, ties, phase/action/card coverage,
  and deterministic fingerprints.
- Reproduce the retained q1 MC control tensor digest exactly. A mismatch blocks dependent claims.

### C. Paired training and development arenas

- Train three MC controls and three ranking treatments from the frozen inputs.
- Run the declared fresh matched arenas sequentially, retaining semantic records and recomputable
  aggregates.

### D. Analysis, independent review, archive, and briefing

- Deterministically recompute arena aggregates and advancement criteria.
- Separate gameplay-block uncertainty from training-replicate variation.
- Obtain a fresh-context review challenging causal attribution, teacher imitation, safety, and
  statistical interpretation.
- Retain a checksum archive, update status, write one results briefing, and stop for user direction.

## Limits

- Wall-clock operating budget: **8 hours** after claim-generating dispatch.
- CPU-only current two-core VM; no GPU, paid service, or new external infrastructure.
- At most two light implementation/analysis jobs and one substantial training/evaluation job; only
  one substantial job runs at a time.
- At most two bounded implementation workers in isolated worktrees and one fresh-context reviewer.
- At most one operational repair-and-retry. A repair may fix implementation/resume/reporting defects
  but may not alter the frozen teacher, loss, corpus, recipe, seeds, arena sizes, or decision rule.
- Generated corpora, datasets, checkpoints, records, and archives remain ignored artifacts; compact
  agreement, plan, results, and source changes are committed.

## Explicitly out of scope

- new self-play or mixed-opponent corpus collection;
- public-history, encoder, model-capacity, action-policy, or terminal-safety changes;
- ranking-loss or optimizer sweeps, adaptive weighting, or post-result seed additions;
- full Gymnasium/PettingZoo environment work;
- promotion, locked-final evaluation, public deployment, or general strength claims; and
- off-VM artifact infrastructure.

## Decisions reserved for the user

A later fresh-corpus confirmation, mixed-opponent replay experiment, locked-final evaluation,
promotion, changed recipe, larger budget, or external retention mechanism requires new approval.
The coordinator must stop after the briefing and cannot launch the next cycle automatically.
